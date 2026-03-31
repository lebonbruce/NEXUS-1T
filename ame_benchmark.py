"""
AME: Attention Memory Engine — 用Transformer原语解决Transformer的遗忘

第一性原理：
  遗忘为什么发生？因为知识存在权重里，学习=修改权重=覆写旧知识。
  那如果知识不存在权重里呢？

  Attention本身就是"不修改权重的即时检索"：
    - KV对 = 记忆
    - Query = 检索
    - attention(Q, K, V) = 在记忆中找到相关内容

  如果每学完一个任务，把关键知识编码为KV对持久化存储，
  推理时让attention自动检索旧记忆——遗忘根本不存在。

  这不是"给Transformer打补丁"，而是用Transformer自己的计算原语
  从根本上解决遗忘问题。
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
from sce.models.base import CLModel, TransformerBlock
from sce.models.naive import NaiveTransformer
from sce.models.replay import ReplayTransformer
from sce.models.progressive import ProgressiveNet


# ============================================================
# AME Block: 标准Attention + FFN + Persistent KV Memory
# ============================================================
class AMEBlock(nn.Module):
    """
    一个Transformer层 + 持久化KV记忆。

    forward pass时：
    1. 标准self-attention处理输入
    2. FFN做非线性变换
    3. Memory cross-attention检索旧知识，补偿FFN的遗忘
    """
    def __init__(self, config, slots_per_task=32):
        super().__init__()
        self.d_model = config.d_model
        self.slots_per_task = slots_per_task

        # 标准Transformer组件
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model))

        # 持久化记忆（不参与梯度计算）
        self._mem_keys = []    # list of (slots, d) tensors
        self._mem_values = []  # list of (slots, d) tensors

        # 可学习的记忆门控（控制记忆影响力）
        self.mem_gate = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1)=0.27

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]

        h2 = self.ln2(x)
        ffn_out = self.ffn(h2)

        if self._mem_keys:
            mem_out = self._read_memory(h2)
            gate = torch.sigmoid(self.mem_gate)
            x = x + (1 - gate) * ffn_out + gate * mem_out
        else:
            x = x + ffn_out
        return x

    def _read_memory(self, query):
        """Cross-attention到持久化记忆——纯Transformer原语操作。"""
        all_k = torch.cat(self._mem_keys, dim=0)   # (M, d)
        all_v = torch.cat(self._mem_values, dim=0)  # (M, d)
        scores = query @ all_k.T / (self.d_model ** 0.5)  # (B, T, M)
        weights = F.softmax(scores, dim=-1)
        return weights @ all_v  # (B, T, d)

    def write_memory(self, ffn_inputs, ffn_outputs):
        """
        无梯度写入：从训练数据中提取代表性KV对。
        K = FFN的输入特征（"什么输入会触发这个记忆"）
        V = FFN的输出特征（"记忆的内容是什么"）
        """
        flat_k = ffn_inputs.reshape(-1, self.d_model).detach()
        flat_v = ffn_outputs.reshape(-1, self.d_model).detach()
        # 按FFN输出的norm选取最重要的slots
        norms = flat_v.norm(dim=-1)
        n = min(self.slots_per_task, len(norms))
        _, idx = norms.topk(n)
        self._mem_keys.append(flat_k[idx].clone())
        self._mem_values.append(flat_v[idx].clone())


# ============================================================
# AME Transformer
# ============================================================
class AMETransformer(CLModel):
    def __init__(self, config, slots_per_task=32):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.blocks = nn.ModuleList([
            AMEBlock(config, slots_per_task) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        self._slots = slots_per_task

    def forward(self, x, task_id):
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

    def on_task_end(self, task_id, train_x, train_y):
        """训练后写入记忆：无梯度的forward pass收集FFN行为。"""
        device = next(self.parameters()).device
        self.eval()
        n = min(500, len(train_x))
        idx = torch.randperm(len(train_x))[:n]
        bx = train_x[idx].to(device)

        B, T = bx.size()
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(bx) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)

        with torch.no_grad():
            for block in self.blocks:
                h_ln = block.ln1(h)
                h = h + block.attn(h_ln, h_ln, h_ln, need_weights=False)[0]
                h_ffn_in = block.ln2(h)
                ffn_out = block.ffn(h_ffn_in)
                block.write_memory(h_ffn_in, ffn_out)
                h = h + ffn_out
        self.train()


# ============================================================
# AME + Replay 组合版本
# ============================================================
class AMEReplayTransformer(AMETransformer):
    """AME + 经验回放：双重记忆保险。"""
    def __init__(self, config, slots_per_task=32):
        super().__init__(config, slots_per_task)
        self._buf_x, self._buf_y, self._buf_t = [], [], []

    def on_task_end(self, task_id, train_x, train_y):
        super().on_task_end(task_id, train_x, train_y)
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
# 大模型Naive（用于效能比对照）
# ============================================================
class LargeNaiveTransformer(CLModel):
    """4倍参数的Naive Transformer，验证'小模型+记忆 vs 大模型'。"""
    def __init__(self, config):
        super().__init__(config)
        d, ff, h = 256, 1024, 8
        self.token_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.seq_len, d)
        self.task_emb = nn.Embedding(config.num_tasks, d)
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            block = nn.ModuleDict({
                'ln1': nn.LayerNorm(d),
                'attn': nn.MultiheadAttention(d, h, batch_first=True),
                'ln2': nn.LayerNorm(d),
                'ffn': nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
            })
            self.blocks.append(block)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.vocab_size)

    def forward(self, x, task_id):
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)
        for block in self.blocks:
            hl = block['ln1'](h)
            h = h + block['attn'](hl, hl, hl, need_weights=False)[0]
            h = h + block['ffn'](block['ln2'](h))
        return self.head(self.ln_f(h))


# ============================================================
# 实验运行器（复用）
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
    pt, pa = model.count_params(True), model.count_params(False)
    # memory slots不算参数但要报告
    mem_size = 0
    if hasattr(model, 'blocks'):
        for b in model.blocks:
            if hasattr(b, '_mem_keys'):
                for mk in b._mem_keys:
                    mem_size += mk.numel() * 2  # keys + values
    print(f" | AA={met['AA']*100:.1f}% BWT={met['BWT']*100:+.1f}% "
          f"T={dur:.0f}s P={pa/1e3:.0f}K Mem={mem_size/1e3:.0f}K")
    return {"name": name, "aa": met["AA"], "bwt": met["BWT"],
            "per_task": acc_matrix[:,-1], "p": pa, "mem": mem_size, "t": dur}


def main():
    config = ExperimentConfig()
    device = torch.device(config.device)
    print("=" * 60)
    print("  AME BENCHMARK: Attention Memory Engine")
    print("=" * 60)

    methods = [
        (NaiveTransformer,     "Naive (812K)"),
        (LargeNaiveTransformer,"Naive-Large (3.2M)"),
        (ReplayTransformer,    "Replay"),
        (ProgressiveNet,       "Progressive"),
        (AMETransformer,       "AME (ours)"),
        (AMEReplayTransformer, "AME+Replay (ours)"),
    ]
    results = []
    for cls, name in methods:
        torch.manual_seed(42); np.random.seed(42)
        results.append(run_method(config, cls, name, device))

    # === 结果表 ===
    print("\n\n" + "=" * 75)
    print(f"{'Method':22s} | {'AA':>6s} | {'BWT':>7s} | {'Params':>7s} | {'Memory':>7s} | {'Time':>5s}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:22s} | {r['aa']*100:5.1f}% | {r['bwt']*100:+5.1f}% | "
              f"{r['p']/1e3:5.0f}K | {r['mem']/1e3:5.0f}K | {r['t']:4.0f}s")
    print("-" * 75)
    print(f"\n{'Per-Task Accuracy':22s} |", end="")
    for tn in TASK_NAMES: print(f" {tn[:7]:>7s}", end="")
    print()
    print("-" * 75)
    for r in results:
        print(f"{r['name']:22s} |", end="")
        for v in r['per_task']: print(f" {v*100:6.1f}%", end="")
        print()

    # === 效能分析 ===
    naive_s = results[0]
    naive_l = results[1]
    replay = results[2]
    prog = results[3]
    ame = results[4]
    ame_r = results[5]

    print("\n" + "=" * 75)
    print("EFFICIENCY ANALYSIS")
    print("=" * 75)
    print(f"Naive-Large has {naive_l['p']/naive_s['p']:.1f}x params of Naive-Small")
    print(f"  Naive-Small AA: {naive_s['aa']*100:.1f}%")
    print(f"  Naive-Large AA: {naive_l['aa']*100:.1f}%")
    print(f"  AME (same size as Small + memory): {ame['aa']*100:.1f}%")
    print(f"  AME+Replay: {ame_r['aa']*100:.1f}%")
    if ame['aa'] > naive_l['aa']:
        ratio = naive_l['p'] / (ame['p'] + ame['mem'])
        print(f"  >> AME beats Naive-Large while using {ratio:.1f}x fewer total resources!")
    if ame_r['aa'] > prog['aa']:
        print(f"  >> AME+Replay ({ame_r['aa']*100:.1f}%) beats Progressive ({prog['aa']*100:.1f}%)!")


if __name__ == "__main__":
    main()
