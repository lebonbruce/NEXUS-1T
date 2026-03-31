"""
TTT-Persistent: TTT + 跨序列持久化W1 + Surprise门控

核心改进（相比ttt_benchmark.py）：
  1. W1跨序列持久化：不再每次forward重新初始化
  2. Surprise门控遗忘：只有"意外"的输入才显著更新W1，防止覆写
  3. Momentum：平滑更新，保留旧知识

这是Titans论文的核心机制，用TTT的代码骨架实现。
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


class PersistentTTTLayer(nn.Module):
    """
    TTT Layer with Persistent Memory (Titans-inspired).
    
    vs 标准TTT:
      标准TTT: 每次forward W1从初始值开始 → 无跨序列记忆
      Persistent: W1跨forward持久化 → 旧任务知识被编码在W1中
    
    vs Titans:
      Titans: 用MLP作为memory，surprise驱动更新
      我们: 用线性模型W1作为memory（更简单），surprise门控更新幅度
    """
    def __init__(self, d_model: int, n_heads: int = 4, ttt_lr: float = 1.0,
                 momentum: float = 0.9, forget_factor: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ttt_base_lr = ttt_lr
        self.momentum = momentum
        self.forget_factor = forget_factor

        # QKV投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # TTT内部线性模型（初始值，可训练）
        self.W1_init = nn.Parameter(
            torch.normal(0, 0.02, size=(n_heads, self.head_dim, self.head_dim)))
        self.b1_init = nn.Parameter(torch.zeros(n_heads, 1, self.head_dim))

        # 持久化状态（不参与梯度，跨forward保留）
        self.register_buffer('W1_state',
            torch.normal(0, 0.02, size=(n_heads, self.head_dim, self.head_dim)))
        self.register_buffer('b1_state',
            torch.zeros(n_heads, 1, self.head_dim))
        # Momentum缓冲
        self.register_buffer('W1_momentum',
            torch.zeros(n_heads, self.head_dim, self.head_dim))
        self.register_buffer('b1_momentum',
            torch.zeros(n_heads, 1, self.head_dim))

        # Surprise门控：控制更新幅度
        self.surprise_proj = nn.Linear(d_model, n_heads, bias=True)

        # 学习率gate
        self.ttt_lr_weight = nn.Parameter(
            torch.normal(0, 0.02, size=(n_heads, d_model, 1)))
        self.ttt_lr_bias = nn.Parameter(torch.zeros(n_heads, 1))

        # TTT内部LayerNorm
        self.ttt_ln_weight = nn.Parameter(torch.ones(n_heads, self.head_dim))
        self.ttt_ln_bias = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.post_norm = nn.LayerNorm(d_model)

        self._initialized = False

    def reset_state(self):
        """重置持久化状态到初始值。"""
        with torch.no_grad():
            self.W1_state.copy_(self.W1_init.data)
            self.b1_state.copy_(self.b1_init.data)
            self.W1_momentum.zero_()
            self.b1_momentum.zero_()
        self._initialized = True

    def _ln_fwd(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / torch.sqrt(var + 1e-6)
        w = self.ttt_ln_weight.view(self.n_heads, 1, self.head_dim)
        b = self.ttt_ln_bias.view(self.n_heads, 1, self.head_dim)
        return w * x_hat + b

    def _ln_fused_l2_bwd(self, x, target):
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
        B, L, _ = hidden_states.shape
        nh, hd = self.n_heads, self.head_dim

        # 初始化持久化状态（首次调用）
        if not self._initialized:
            self.reset_state()

        # QKV投影
        XQ = self.q_proj(hidden_states).view(B, L, nh, hd).permute(0, 2, 1, 3)
        XK = self.k_proj(hidden_states).view(B, L, nh, hd).permute(0, 2, 1, 3)
        XV = self.v_proj(hidden_states).view(B, L, nh, hd).permute(0, 2, 1, 3)

        # === 使用持久化W1（不是每次从init开始） ===
        W1 = self.W1_state.unsqueeze(0).expand(B, -1, -1, -1)
        b1 = self.b1_state.unsqueeze(0).expand(B, -1, -1, -1)

        # 学习率eta
        ttt_lr = torch.einsum('bld,hdo->bhlo', hidden_states, self.ttt_lr_weight)
        ttt_lr = ttt_lr + self.ttt_lr_bias.view(1, nh, 1, 1)
        ttt_lr = torch.sigmoid(ttt_lr) * self.ttt_base_lr / hd

        # Surprise门控：计算当前输入的"意外程度"
        # surprise越高 → 更新幅度越大
        surprise = torch.sigmoid(self.surprise_proj(hidden_states))  # [B, L, nh]
        surprise = surprise.permute(0, 2, 1).unsqueeze(-1)  # [B, nh, L, 1]
        # 调制学习率
        ttt_lr = ttt_lr * surprise

        # TTT dual form计算
        Z1 = XK @ W1 + b1
        reconstruction_target = XV - XK
        grad_l_wrt_Z1 = self._ln_fused_l2_bwd(Z1, reconstruction_target)

        Attn1 = torch.tril(XQ @ XK.transpose(-2, -1))
        eta_tril = torch.tril(ttt_lr.expand(-1, -1, -1, L))

        b1_bar = b1 - eta_tril @ grad_l_wrt_Z1
        Z1_bar = XQ @ W1 - (ttt_lr * Attn1) @ grad_l_wrt_Z1 + b1_bar
        Z1_bar = self._ln_fwd(Z1_bar)
        output = XQ + Z1_bar

        # === 持久化更新W1（Titans的核心：跨序列记忆） ===
        if self.training:
            with torch.no_grad():
                # 计算此次forward的W1更新量（取最后一个token的梯度）
                last_eta = ttt_lr[:, :, -1:, :]  # [B, nh, 1, 1]
                # W1的梯度：K^T @ grad
                dW1 = (last_eta * XK[:, :, -1:, :]).transpose(-2, -1) @ grad_l_wrt_Z1[:, :, -1:, :]
                db1 = last_eta * grad_l_wrt_Z1[:, :, -1:, :]

                # 对batch取平均
                dW1 = dW1.mean(0)  # [nh, hd, hd]
                db1 = db1.mean(0)  # [nh, 1, hd]

                # Momentum更新（平滑，防止剧烈覆写）
                self.W1_momentum.mul_(self.momentum).add_(dW1, alpha=1-self.momentum)
                self.b1_momentum.mul_(self.momentum).add_(db1, alpha=1-self.momentum)

                # 应用更新 + 轻微遗忘（防止W1无限增长）
                self.W1_state.mul_(1 - self.forget_factor).sub_(self.W1_momentum)
                self.b1_state.mul_(1 - self.forget_factor).sub_(self.b1_momentum)

        output = output.permute(0, 2, 1, 3).reshape(B, L, self.d_model)
        output = self.post_norm(output)
        return self.o_proj(output)


class PersistentTTTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ttt = PersistentTTTLayer(config.d_model, config.n_heads)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff), nn.GELU(),
            nn.Linear(config.d_ff, config.d_model))

    def forward(self, x):
        x = x + self.ttt(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class PersistentTTTModel(CLModel):
    def __init__(self, config):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.blocks = nn.ModuleList([PersistentTTTBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, task_id):
        B, T = x.size()
        device = x.device
        h = self.token_emb(x) + self.pos_emb(torch.arange(T, device=device)) + \
            self.task_emb(torch.full((B,), task_id, dtype=torch.long, device=device)).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

    def on_task_start(self, task_id):
        if task_id == 0:
            for block in self.blocks:
                block.ttt.reset_state()


class PersistentTTTReplay(PersistentTTTModel):
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
# 运行器
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
    print("=" * 65)
    print("  TTT-PERSISTENT: Titans-style Persistent Memory for CL")
    print("=" * 65)

    from ttt_benchmark import TTTTransformer, TTTReplayTransformer

    methods = [
        (NaiveTransformer,       "Naive"),
        (ReplayTransformer,      "Replay"),
        (ProgressiveNet,         "Progressive"),
        (TTTReplayTransformer,   "TTT+Replay (no persist)"),
        (PersistentTTTModel,     "TTT-Persistent"),
        (PersistentTTTReplay,    "TTT-Persistent+Replay"),
    ]
    results = []
    for cls, name in methods:
        torch.manual_seed(42); np.random.seed(42)
        results.append(run_method(config, cls, name, device))

    print("\n\n" + "=" * 70)
    print(f"{'Method':26s} | {'AA':>6s} | {'BWT':>7s} | {'Params':>7s} | {'Time':>5s}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:26s} | {r['aa']*100:5.1f}% | {r['bwt']*100:+5.1f}% | "
              f"{r['p']/1e3:5.0f}K | {r['t']:4.0f}s")
    print("-" * 70)
    print(f"\n{'Per-Task':26s} |", end="")
    for tn in TASK_NAMES: print(f" {tn[:7]:>7s}", end="")
    print()
    for r in results:
        print(f"{r['name']:26s} |", end="")
        for v in r['per_task']: print(f" {v*100:6.1f}%", end="")
        print()


if __name__ == "__main__":
    main()
