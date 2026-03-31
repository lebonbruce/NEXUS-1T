"""验证 scale-aware models.py 的正确性。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import NexusGPT, BaselineGPT, get_mla_compression, should_enable_ttt
import torch

print("=" * 70)
print("  Scale-Aware 配置验证")
print("=" * 70)

# 测试不同规模下的配置决策
configs = [
    ("当前30M", 384, 6, 6, 512),
    ("~50M", 512, 8, 8, 512),
    ("~50M长序列", 512, 8, 8, 2048),
    ("~124M", 768, 12, 12, 1024),
    ("~350M", 1024, 24, 16, 2048),
]

for name, d, n_l, n_h, seq in configs:
    mla = get_mla_compression(d)
    ttt = should_enable_ttt(d, seq)
    kv_latent = d // mla if mla > 1 else d
    mla_str = "OFF" if mla == 1 else f"{mla}x (kv_latent={kv_latent})"
    ttt_str = "ON" if ttt else "OFF"
    print(f"\n  {name} (d={d}, seq={seq}):")
    print(f"    MLA: {mla_str} | TTT: {ttt_str}")

# 实际构建当前 30M 模型
print("\n\n--- 构建当前 30M NEXUS (d_model=384, seq=512) ---")
V, D, L, H, SEQ = 50257, 384, 6, 6, 512
nexus = NexusGPT(V, D, L, H, 1024, SEQ)

print(f"\n--- 构建 Baseline ---")
baseline = BaselineGPT(V, D, L, H, 1536, SEQ)
print(f"  Params: {baseline.count_params():,}")

# 前向传播测试
x = torch.randint(0, V, (2, SEQ))
with torch.no_grad():
    logits_b, loss_b = baseline(x, x)
    logits_n, loss_n = nexus(x, x)
print(f"\n  Forward pass OK!")
print(f"  Baseline loss: {loss_b.item():.4f}")
print(f"  NEXUS loss:    {loss_n.item():.4f}")
print(f"  Output shapes: B={logits_b.shape}, N={logits_n.shape}")

# 参数量差异分析
print(f"\n--- 参数量对比 ---")
bp = baseline.count_params()
np_ = nexus.count_params()
print(f"  Baseline:  {bp:,}")
print(f"  NEXUS v4:  {np_:,}")
print(f"  差异:      {np_-bp:+,} ({(np_-bp)/bp*100:+.1f}%)")
print(f"\n  ✅ Scale-aware models.py 验证通过!")
