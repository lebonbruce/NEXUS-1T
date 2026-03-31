"""
消融实验：逐个关闭 NEXUS 组件，定位 loss 差距的来源。

实验设计（1000 步快速验证，足以看出趋势）：
  1. Baseline:       MHA + GELU               → 已有结果 val=4.08
  2. NEXUS Full:     DiffAttn+MLA + TTT + SwiGLU → 已有结果 val=4.45
  3. NEXUS no-TTT:   DiffAttn+MLA + SwiGLU      → TTT 是否是拖累？
  4. Baseline+SwiGLU: MHA + SwiGLU              → SwiGLU 单独效果？
  5. DiffAttn Only:  DiffAttn+MLA + GELU        → DiffAttn 单独效果？
"""
import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    RMSNorm, CausalSelfAttention, DiffAttnMLA,
    GELUMLP, SwiGLUFFN, TTTLinear,
    BaselineGPT, NexusGPT
)
import torch.nn as nn


# ============================================================
# 消融变体模型
# ============================================================

class NexusNoTTTBlock(nn.Module):
    """NEXUS 去掉 TTT：DiffAttn+MLA + SwiGLU（无 TTT）"""
    def __init__(self, d_model, n_heads, d_ff, seq_len, layer_idx, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = DiffAttnMLA(d_model, n_heads, seq_len, layer_idx, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineSwiGLUBlock(nn.Module):
    """Baseline + SwiGLU：标准 MHA + SwiGLU（替换 GELU）"""
    def __init__(self, d_model, n_heads, d_ff, seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DiffAttnOnlyBlock(nn.Module):
    """DiffAttn Only：DiffAttn+MLA + GELU（无 TTT, 无 SwiGLU）"""
    def __init__(self, d_model, n_heads, d_ff, seq_len, layer_idx, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = DiffAttnMLA(d_model, n_heads, seq_len, layer_idx, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = GELUMLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class AblationGPT(nn.Module):
    """通用消融模型，支持不同 Block 类型。"""
    def __init__(self, block_cls, vocab_size, d_model, n_layers, n_heads,
                 d_ff, seq_len, needs_layer_idx=False, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        blocks = []
        for i in range(n_layers):
            if needs_layer_idx:
                blocks.append(block_cls(d_model, n_heads, d_ff, seq_len, i, dropout))
            else:
                blocks.append(block_cls(d_model, n_heads, d_ff, seq_len, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        h = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 训练函数（简化版，1000 步快速验证）
# ============================================================

def load_data(data_dir):
    """加载已缓存的 memmap 数据。"""
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    return train_data, val_data


def get_batch(data, batch_size, seq_len, device):
    ix = np.random.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.from_numpy(np.stack([data[i:i+seq_len].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i+1:i+seq_len+1].astype(np.int64) for i in ix]))
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, val_data, batch_size, seq_len, device, n_eval=10):
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(val_data, batch_size, seq_len, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train_ablation(name, model, train_data, val_data, max_steps=1000,
                   batch_size=8, seq_len=512, grad_accum=4, device='cuda'):
    """训练一个消融变体。"""
    model = model.to(device)
    params = model.count_params()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95),
                                  weight_decay=0.1)

    print(f"\n{'='*60}")
    print(f"  消融: {name}")
    print(f"  参数量: {params:,}")
    print(f"{'='*60}")

    t0 = time.time()
    val_history = []

    for step in range(max_steps):
        # 学习率调度（warmup + cosine）
        if step < 200:
            lr = 6e-4 * step / 200
        else:
            progress = (step - 200) / (max_steps - 200)
            lr = 6e-4 * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 梯度累积
        optimizer.zero_grad()
        total_loss = 0
        for _ in range(grad_accum):
            x, y = get_batch(train_data, batch_size, seq_len, device)
            _, loss = model(x, y)
            (loss / grad_accum).backward()
            total_loss += loss.item()
        total_loss /= grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 250 == 0 and step > 0:
            vl = eval_loss(model, val_data, batch_size, seq_len, device)
            val_history.append((step, vl))
            elapsed = time.time() - t0
            print(f"  [{name}] step {step:4d} | val_loss {vl:.4f} | {elapsed:.0f}s")

        if step % 100 == 0:
            elapsed = time.time() - t0
            tok_per_s = (step + 1) * batch_size * seq_len * grad_accum / max(elapsed, 1)
            print(f"  [{name}] step {step:4d}/{max_steps} | loss {total_loss:.4f} | "
                  f"{tok_per_s:.0f} tok/s | {elapsed:.0f}s")

    # 最终 val
    final_val = eval_loss(model, val_data, batch_size, seq_len, device, n_eval=20)
    total_time = time.time() - t0
    val_history.append((max_steps, final_val))

    print(f"\n  [{name}] 完成! val_loss={final_val:.4f} | {total_time:.0f}s")

    del model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "params": params,
        "final_val_loss": final_val,
        "time_s": total_time,
        "val_history": val_history,
    }


# ============================================================
# 主程序
# ============================================================

def main():
    device = 'cuda'
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    train_data, val_data = load_data(data_dir)
    print(f"数据: train={len(train_data):,} val={len(val_data):,} tokens")

    # 模型配置
    V, D, L, H, SEQ = 50257, 384, 6, 6, 512

    variants = [
        # (名称, 模型, 预计时间)
        ("Baseline+SwiGLU",
         AblationGPT(BaselineSwiGLUBlock, V, D, L, H, 1024, SEQ, needs_layer_idx=False),
         "~2 min"),
        ("DiffAttn+GELU",
         AblationGPT(DiffAttnOnlyBlock, V, D, L, H, 1536, SEQ, needs_layer_idx=True),
         "~2 min"),
        ("NEXUS-noTTT",
         AblationGPT(NexusNoTTTBlock, V, D, L, H, 1024, SEQ, needs_layer_idx=True),
         "~2 min"),
    ]

    print(f"\n{'='*60}")
    print(f"  NEXUS 消融实验（1000 步快速验证）")
    print(f"  已有结果: Baseline val=4.08, NEXUS-Full val=4.45")
    print(f"  待测变体: {len(variants)} 个")
    print(f"{'='*60}")

    results = []
    # 添加已有结果
    results.append({"name": "Baseline (MHA+GELU)", "params": 29920512,
                     "final_val_loss": 4.0801, "time_s": 2417, "note": "3000步已有结果"})
    results.append({"name": "NEXUS Full", "params": 32363904,
                     "final_val_loss": 4.4481, "time_s": 18718, "note": "3000步已有结果"})

    for name, model, est_time in variants:
        print(f"\n  下一个: {name} (预计 {est_time})")
        torch.manual_seed(42)
        result = train_ablation(name, model, train_data, val_data,
                                max_steps=1000, device=device)
        results.append(result)

    # 打印汇总
    print(f"\n\n{'='*60}")
    print(f"  消融实验结果汇总")
    print(f"{'='*60}\n")
    print(f"  {'Model':<25s} | {'Params':>10s} | {'Val Loss':>10s} | {'Time':>8s}")
    print(f"  {'-'*60}")
    for r in results:
        t = f"{r['time_s']:.0f}s" if r['time_s'] < 600 else f"{r['time_s']/60:.1f}m"
        print(f"  {r['name']:<25s} | {r['params']:>10,} | {r['final_val_loss']:>10.4f} | {t:>8s}")

    # 保存结果
    out_path = os.path.join(data_dir, "ablation_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
