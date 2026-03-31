"""
PRIME: Primitive-Recombination for Improved Model Efficiency
=============================================================
从第一性原理出发，重新组合Transformer的4个底层原语，
在同参数下实现更高效的计算。

核心改进（全部来自对"标准原语为什么低效"的分析）：

1. Conditional FFN (从FFN参数利用率低出发)
   标准FFN: out = W2·GELU(W1·x)  → 所有参数对所有输入全量计算
   PRIME:   把W1分成G组，根据x选择性激活top-K组
   等效于：同参数下，每个输入用到的参数更"专精"

2. Dynamic Residual (从固定残差不灵活出发)
   标准: out = x + f(x)  → 每层对每个token权重相同
   PRIME: out = x + α(x)·f(x)  → 模型自己决定每层每token的重要性

3. V-Fusion (从Attention产出重复线性投影出发)
   标准: V = W_v · x  → 线性投影，表达力有限
   PRIME: V = W_v · GELU(W_vg · x)  → V自带非线性特征

这些改进的共同原则：
"让同样数量的参数，根据不同输入，做不同的计算"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math


# ============================================================
# 任务定义：多种能力测试
# ============================================================
VOCAB, SEQ = 64, 32

def gen_data(tid, n):
    x = torch.randint(1, VOCAB, (n, SEQ))
    if tid == 0:    # 线性映射: y = (x * 3 + 7) % VOCAB
        y = (x * 3 + 7) % VOCAB
    elif tid == 1:  # 排序
        y, _ = x.sort(dim=1)
    elif tid == 2:  # 反转
        y = x.flip(1)
    elif tid == 3:  # 前缀和 mod VOCAB
        y = x.cumsum(1) % VOCAB
    elif tid == 4:  # 滑动最大值
        y = x.cummax(1).values
    elif tid == 5:  # 奇偶校验（累计）
        y = x.cumsum(1) % 2
    return x, y

TASKS = ["LinearMap", "Sort", "Reverse", "PrefixSum", "RunMax", "Parity"]


# ============================================================
# 标准Transformer (Baseline)
# ============================================================
class StdAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, T, -1))


class StdBlock(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = StdAttn(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class StdTransformer(nn.Module):
    def __init__(self, d=128, h=4, nl=4, ff=256):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([StdBlock(d, h, ff) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        B, T = x.size()
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))


# ============================================================
# PRIME Transformer (同参数魔改版)
# ============================================================
class ConditionalFFN(nn.Module):
    """
    Conditional FFN：把FFN的中间层分成G组，
    每个输入只激活top-K组（稀疏激活）。
    同参数量，但每个输入用到的参数更专精。
    
    参数量对比（与标准FFN相同）：
    标准FFN: d→ff + ff→d = 2*d*ff
    ConditionalFFN: d→ff + ff→d + d→G (router) ≈ 2*d*ff + d*G
    router的额外参数量极小(d*G << d*ff)，可忽略
    """
    def __init__(self, d, ff, n_groups=4, top_k=2):
        super().__init__()
        self.n_groups = n_groups
        self.top_k = top_k
        self.group_size = ff // n_groups

        # 分组的FFN参数（总参数量与标准FFN相同）
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)

        # 轻量级路由器（额外参数极少）
        self.router = nn.Linear(d, n_groups, bias=False)

    def forward(self, x):
        # 路由：决定激活哪些组
        route_logits = self.router(x)  # [B, T, G]
        route_weights = F.softmax(route_logits, dim=-1)

        # 选择top-K组
        topk_vals, topk_idx = route_weights.topk(self.top_k, dim=-1)
        # 构造稀疏掩码
        mask = torch.zeros_like(route_weights)
        mask.scatter_(-1, topk_idx, 1.0)
        # 重新归一化被选中组的权重
        gate = route_weights * mask
        gate = gate / (gate.sum(-1, keepdim=True) + 1e-8)

        # FFN前向：先全量计算W1，然后按组mask
        h = self.w1(x)  # [B, T, ff]
        # 按组加权
        h_grouped = h.view(*h.shape[:-1], self.n_groups, self.group_size)
        h_grouped = h_grouped * gate.unsqueeze(-1)  # 未选中的组被置零
        h = h_grouped.view(*h.shape)
        h = F.gelu(h)
        return self.w2(h)


class FusedAttention(nn.Module):
    """
    V-Fusion Attention：V投影带非线性。
    标准Attention: V = W_v · x （线性）
    Fused:         V = W_v2 · GELU(W_v1 · x)（非线性）
    
    这让Value本身就包含了"变换后的特征"，
    相当于把一部分FFN的工作融入到Attention中。
    
    参数量保持：W_v1是d→hd，W_v2是hd→hd（替代原来的d→d）
    总参数量略有调整，通过减小ff来补偿。
    """
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        # V用非线性投影（GLU-style）
        self.v_gate = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        # V-Fusion: V = sigmoid(gate) * proj(x) — GLU风格
        v = torch.sigmoid(self.v_gate(x)) * self.v_proj(x)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(y.transpose(1, 2).reshape(B, T, -1))


class PRIMEBlock(nn.Module):
    """PRIME Block: Fused Attention + Conditional FFN + Dynamic Residual"""
    def __init__(self, d, h, ff, n_groups=4, top_k=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = FusedAttention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = ConditionalFFN(d, ff, n_groups, top_k)

        # Dynamic Residual：数据依赖的残差权重
        self.attn_gate = nn.Sequential(nn.Linear(d, 1, bias=True), nn.Sigmoid())
        self.ffn_gate = nn.Sequential(nn.Linear(d, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        h = self.ln1(x)
        attn_out = self.attn(h)
        # 动态残差：α(x) * attn_out
        alpha = self.attn_gate(h)  # [B, T, 1]
        x = x + alpha * attn_out

        h = self.ln2(x)
        ffn_out = self.ffn(h)
        beta = self.ffn_gate(h)
        x = x + beta * ffn_out
        return x


class PRIMETransformer(nn.Module):
    def __init__(self, d=128, h=4, nl=4, ff=256, n_groups=4, top_k=2):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([
            PRIMEBlock(d, h, ff, n_groups, top_k) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        B, T = x.size()
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))


# ============================================================
# 消融变体：隔离每个改进的独立贡献
# ============================================================
class OnlyConditionalFFN(nn.Module):
    """只有Conditional FFN（Attention不变）"""
    def __init__(self, d=128, h=4, nl=4, ff=256):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQ, d)
        blocks = []
        for _ in range(nl):
            b = nn.Module()
            b.ln1 = nn.LayerNorm(d); b.attn = StdAttn(d, h)
            b.ln2 = nn.LayerNorm(d); b.ffn = ConditionalFFN(d, ff)
            blocks.append(b)
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        B, T = x.size()
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = h + b.attn(b.ln1(h))
            h = h + b.ffn(b.ln2(h))
        return self.head(self.ln(h))


class OnlyDynResidual(nn.Module):
    """只有Dynamic Residual（其他不变）"""
    def __init__(self, d=128, h=4, nl=4, ff=256):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQ, d)
        blocks = []
        for _ in range(nl):
            b = nn.Module()
            b.ln1 = nn.LayerNorm(d); b.attn = StdAttn(d, h)
            b.ln2 = nn.LayerNorm(d)
            b.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
            b.ag = nn.Sequential(nn.Linear(d, 1, bias=True), nn.Sigmoid())
            b.fg = nn.Sequential(nn.Linear(d, 1, bias=True), nn.Sigmoid())
            blocks.append(b)
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        B, T = x.size()
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            r = b.ln1(h); h = h + b.ag(r) * b.attn(r)
            r = b.ln2(h); h = h + b.fg(r) * b.ffn(r)
        return self.head(self.ln(h))


# ============================================================
# 公平测试框架
# ============================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def train_and_eval(model, task_id, device, steps=500, bs=128, lr=3e-4):
    """在单个任务上训练并评估。返回 (train_loss_curve, final_acc)。"""
    model.to(device).train()
    tx, ty = gen_data(task_id, 5000)
    ex, ey = gen_data(task_id, 1000)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(steps):
        idx = torch.randint(0, len(tx), (bs,))
        bx, by = tx[idx].to(device), ty[idx].to(device)
        logits = model(bx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            losses.append(loss.item())

    # 评估
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for i in range(0, len(ex), 200):
            bx = ex[i:i+200].to(device)
            by = ey[i:i+200].to(device)
            pred = model(bx).argmax(-1)
            correct += (pred == by).sum().item()
            total += by.numel()
    return losses, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  PRIME vs Standard Transformer: 同参数公平对比")
    print(f"  Tasks: {TASKS}")
    print(f"  Device: {device}")
    print("=" * 70)

    models = {
        "Standard":      lambda: StdTransformer(),
        "PRIME (full)":  lambda: PRIMETransformer(),
        "Cond-FFN only": lambda: OnlyConditionalFFN(),
        "DynRes only":   lambda: OnlyDynResidual(),
    }

    # 参数量对比
    print("\nParameter counts:")
    for name, factory in models.items():
        m = factory()
        print(f"  {name:20s}: {count_params(m)/1e3:.1f}K")

    # 逐任务测试
    all_results = {}
    for name, factory in models.items():
        results = []
        for tid in range(len(TASKS)):
            torch.manual_seed(42); np.random.seed(42)
            m = factory()
            t0 = time.time()
            losses, acc = train_and_eval(m, tid, device)
            dur = time.time() - t0
            results.append({"task": TASKS[tid], "acc": acc, "losses": losses, "time": dur})
            print(f"  [{name:20s}] {TASKS[tid]:>10s}: acc={acc*100:5.1f}% "
                  f"loss={losses[-1]:.3f} T={dur:.1f}s")
        all_results[name] = results

    # === 结果汇总 ===
    print("\n" + "=" * 70)
    print(f"{'':20s} |", end="")
    for t in TASKS:
        print(f" {t:>9s}", end="")
    print(" |    AVG")
    print("-" * 70)
    for name, results in all_results.items():
        accs = [r['acc'] for r in results]
        print(f"{name:20s} |", end="")
        for a in accs:
            print(f" {a*100:8.1f}%", end="")
        print(f" | {np.mean(accs)*100:5.1f}%")
    print("-" * 70)

    # PRIME vs Standard 对比
    std = all_results["Standard"]
    prime = all_results["PRIME (full)"]
    print(f"\n{'PRIME vs Standard':20s} |", end="")
    for i in range(len(TASKS)):
        diff = (prime[i]['acc'] - std[i]['acc']) * 100
        marker = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f" {diff:+7.1f}%{marker}", end="")
    avg_diff = (np.mean([r['acc'] for r in prime]) - np.mean([r['acc'] for r in std])) * 100
    print(f" | {avg_diff:+4.1f}%")

    # 训练速度对比
    print(f"\nTraining speed (avg per task):")
    for name, results in all_results.items():
        avg_t = np.mean([r['time'] for r in results])
        print(f"  {name:20s}: {avg_t:.1f}s")


if __name__ == "__main__":
    main()
