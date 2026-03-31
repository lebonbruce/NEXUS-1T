"""
TTT-CL: 用Test-Time Training机制实现持续学习

核心原理（从TTT官方代码提取并简化）：
  传统Transformer层: input → Attention(Q,K,V) → FFN → output
  TTT层: input → TTT_self_supervised_update → output

  TTT的hidden state是一个线性模型W，在forward pass中：
  1. 用K作为输入，V作为目标，做一步自监督学习更新W
  2. 用更新后的W(Q)作为输出
  
  这意味着模型在推理时也在学习！
  每处理一个token，W就被更新一次。

  用于持续学习时：
  - W的更新跨序列持久化 → 旧任务的知识被编码在W中
  - 新任务的token流过时，W继续被更新 → 学习新知识
  - W的容量有限 → 需要与其他机制配合防止遗忘

参考：
  - TTT Official: https://github.com/test-time-training/ttt-lm-pytorch
  - Titans: https://github.com/lucidrains/titans-pytorch
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from sce.config import ExperimentConfig
from sce.tasks import generate_task_data, TASK_NAMES
from sce.evaluation import compute_accuracy, compute_cl_metrics
from sce.models.base import CLModel
from sce.models.naive import NaiveTransformer
from sce.models.replay import ReplayTransformer
from sce.models.progressive import ProgressiveNet


# ============================================================
# TTT Layer (从官方ttt.py简化提取)
# ============================================================
class TTTLinearLayer(nn.Module):
    """
    Test-Time Training Linear Layer.
    
    核心：hidden state W是一个线性模型，在forward中通过
    自监督学习(重建V from K)来更新。
    
    简化版：去掉了RoPE、Conv、Gate等工程优化，
    保留纯粹的TTT学习机制。
    """
    def __init__(self, d_model: int, n_heads: int = 4, ttt_lr: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ttt_base_lr = ttt_lr

        # QKV投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # TTT内部线性模型 W1, b1（即"hidden state"）
        # 每个head一个独立的线性模型
        self.W1 = nn.Parameter(
            torch.normal(0, 0.02, size=(n_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(n_heads, 1, self.head_dim))

        # 可学习的learning rate gate（来自TTT论文Sec 2.7）
        self.ttt_lr_weight = nn.Parameter(
            torch.normal(0, 0.02, size=(n_heads, d_model, 1)))
        self.ttt_lr_bias = nn.Parameter(torch.zeros(n_heads, 1))

        # TTT内部LayerNorm参数
        self.ttt_ln_weight = nn.Parameter(torch.ones(n_heads, self.head_dim))
        self.ttt_ln_bias = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        self.post_norm = nn.LayerNorm(d_model)

    def _ln_fwd(self, x):
        """Per-head LayerNorm forward."""
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / torch.sqrt(var + 1e-6)
        w = self.ttt_ln_weight.view(self.n_heads, 1, self.head_dim)
        b = self.ttt_ln_bias.view(self.n_heads, 1, self.head_dim)
        return w * x_hat + b

    def _ln_fused_l2_bwd(self, x, target):
        """LayerNorm + L2 loss的梯度（TTT核心：自监督学习信号）。"""
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-6)
        x_hat = (x - mu) / std
        D = x.shape[-1]
        
        w = self.ttt_ln_weight.view(self.n_heads, 1, self.head_dim)
        b = self.ttt_ln_bias.view(self.n_heads, 1, self.head_dim)
        y = w * x_hat + b
        
        grad_output = y - target
        grad_x_hat = grad_output * w
        z = (1.0 / D) * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        ) / std
        return z

    def forward(self, hidden_states: torch.Tensor):
        """
        TTT Forward Pass:
        1. 投影得到Q, K, V
        2. 用K作为输入、V-K作为目标，计算自监督梯度
        3. 用dual form高效更新W1并计算输出
        """
        B, L, _ = hidden_states.shape
        nh, hd = self.n_heads, self.head_dim

        # QKV投影并reshape为多头
        XQ = self.q_proj(hidden_states).view(B, L, nh, hd).permute(0, 2, 1, 3)
        XK = self.k_proj(hidden_states).view(B, L, nh, hd).permute(0, 2, 1, 3)
        XV = self.v_proj(hidden_states).view(B, L, nh, hd).permute(0, 2, 1, 3)
        # 现在 XQ/XK/XV: [B, nh, L, hd]

        # 计算学习率 eta（数据依赖的）
        # hidden_states: [B, L, d] -> ttt_lr: [B, nh, L, 1]
        ttt_lr = torch.einsum('bld,hdo->bhlo', hidden_states, self.ttt_lr_weight)
        ttt_lr = ttt_lr + self.ttt_lr_bias.view(1, nh, 1, 1)
        ttt_lr = torch.sigmoid(ttt_lr) * self.ttt_base_lr / hd

        # --- TTT核心：Dual Form自监督更新 ---
        # W1初始化：广播到batch维度
        W1 = self.W1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, nh, hd, hd]
        b1 = self.b1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, nh, 1, hd]

        # Step 1: 前向：Z1 = K @ W1 + b1
        Z1 = XK @ W1 + b1  # [B, nh, L, hd]

        # Step 2: 自监督目标：重建 V-K
        reconstruction_target = XV - XK  # [B, nh, L, hd]

        # Step 3: 计算梯度 (LayerNorm + L2)
        grad_l_wrt_Z1 = self._ln_fused_l2_bwd(Z1, reconstruction_target)

        # Step 4: Dual form计算（高效的在线学习）
        # Attn1 = tril(Q @ K^T)，因果掩码确保只看到之前的token
        Attn1 = torch.tril(XQ @ XK.transpose(-2, -1))  # [B, nh, L, L]

        # eta_tril: 学习率的累积效应
        eta_tril = torch.tril(ttt_lr.expand(-1, -1, -1, L))  # [B, nh, L, L]

        # 更新后的输出（dual form等价于逐token SGD更新W1）
        # Z1_bar = Q @ W1 - (eta * Attn1) @ grad + (b1 - eta_tril @ grad)
        b1_bar = b1 - eta_tril @ grad_l_wrt_Z1  # [B, nh, L, hd]
        Z1_bar = XQ @ W1 - (ttt_lr * Attn1) @ grad_l_wrt_Z1 + b1_bar

        # LayerNorm + 残差
        Z1_bar = self._ln_fwd(Z1_bar)
        output = XQ + Z1_bar  # 残差连接

        # reshape回 [B, L, d]
        output = output.permute(0, 2, 1, 3).reshape(B, L, self.d_model)
        output = self.post_norm(output)
        output = self.o_proj(output)

        return output


# ============================================================
# TTT Block: TTT Layer + FFN
# ============================================================
class TTTBlock(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ttt = TTTLinearLayer(config.d_model, config.n_heads, ttt_lr=1.0)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model))

    def forward(self, x):
        x = x + self.ttt(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# TTT Transformer (用于CL benchmark)
# ============================================================
class TTTTransformer(CLModel):
    """TTT用于持续学习：用test-time learning替代标准attention。"""
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.blocks = nn.ModuleList([TTTBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, task_id):
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


# ============================================================
# TTT + Replay
# ============================================================
class TTTReplayTransformer(TTTTransformer):
    def __init__(self, config):
        super().__init__(config)
        self._buf_x, self._buf_y, self._buf_t = [], [], []

    def on_task_end(self, task_id, train_x, train_y):
        n = min(200, len(train_x))
        idx = torch.randperm(len(train_x))[:n]
        self._buf_x.append(train_x[idx].cpu())
        self._buf_y.append(train_y[idx].cpu())
        self._buf_t.append(task_id)

    def get_replay_data(self, batch_size):
        if not self._buf_x:
            return None
        bi = torch.randint(0, len(self._buf_x), (1,)).item()
        n = min(batch_size, len(self._buf_x[bi]))
        si = torch.randint(0, len(self._buf_x[bi]), (n,))
        dev = next(self.parameters()).device
        return self._buf_x[bi][si].to(dev), self._buf_y[bi][si].to(dev), self._buf_t[bi]


# ============================================================
# 实验运行器
# ============================================================
def run_method(config, model_class, name, device):
    print(f"\n  [{name}]", end="", flush=True)
    test_data = [generate_task_data(t, config.test_samples, config.vocab_size, config.seq_len)
                 for t in range(config.num_tasks)]
    model = model_class(config).to(device)
    acc_matrix = np.zeros((config.num_tasks, config.num_tasks))
    t0 = time.time()

    for tid in range(config.num_tasks):
        tx, ty = generate_task_data(tid, config.train_samples, config.vocab_size, config.seq_len)
        model.on_task_start(tid)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable:
            opt = torch.optim.Adam(trainable, lr=config.lr)
            model.train()
            for step in range(config.steps_per_task):
                idx = torch.randint(0, len(tx), (config.batch_size,))
                bx, by = tx[idx].to(device), ty[idx].to(device)
                logits = model(bx, tid)
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), by.reshape(-1))
                loss = loss + model.compute_extra_loss()
                replay = model.get_replay_data(config.batch_size // 2)
                if replay:
                    rx, ry, rt = replay
                    loss = loss + F.cross_entropy(
                        model(rx, rt).reshape(-1, config.vocab_size), ry.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
        model.on_task_end(tid, tx, ty)
        for eid in range(config.num_tasks):
            acc_matrix[eid, tid] = compute_accuracy(model, *test_data[eid], eid, device)
        print(f" T{tid}:{acc_matrix[tid,tid]*100:.0f}%", end="", flush=True)

    dur = time.time() - t0
    met = compute_cl_metrics(acc_matrix)
    pa = model.count_params(False)
    print(f" | AA={met['AA']*100:.1f}% BWT={met['BWT']*100:+.1f}% T={dur:.0f}s P={pa/1e3:.0f}K")
    return {"name": name, "aa": met["AA"], "bwt": met["BWT"],
            "per_task": acc_matrix[:,-1], "p": pa, "t": dur}


def main():
    config = ExperimentConfig()
    device = torch.device(config.device)
    print("=" * 60)
    print("  TTT-CL BENCHMARK")
    print("  Test-Time Training for Continual Learning")
    print("=" * 60)

    methods = [
        (NaiveTransformer,      "Naive"),
        (ReplayTransformer,     "Replay"),
        (ProgressiveNet,        "Progressive"),
        (TTTTransformer,        "TTT-Linear"),
        (TTTReplayTransformer,  "TTT-Linear+Replay"),
    ]
    results = []
    for cls, name in methods:
        torch.manual_seed(42); np.random.seed(42)
        results.append(run_method(config, cls, name, device))

    # 结果表
    print("\n\n" + "=" * 70)
    print(f"{'Method':22s} | {'AA':>6s} | {'BWT':>7s} | {'Params':>7s} | {'Time':>5s}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:22s} | {r['aa']*100:5.1f}% | {r['bwt']*100:+5.1f}% | "
              f"{r['p']/1e3:5.0f}K | {r['t']:4.0f}s")
    print("-" * 70)
    print(f"\n{'Per-Task':22s} |", end="")
    for tn in TASK_NAMES: print(f" {tn[:7]:>7s}", end="")
    print()
    for r in results:
        print(f"{r['name']:22s} |", end="")
        for v in r['per_task']: print(f" {v*100:6.1f}%", end="")
        print()


if __name__ == "__main__":
    main()
