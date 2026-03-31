"""50M 模型完整验证（LoRA rank=64 重构后）"""
import sys; sys.path.insert(0, 'pretrain')
import torch, torch.nn.functional as F, numpy as np, time, math
from models import NexusGPT, BaselineGPT

t0 = time.time()
D, V, L, H = 512, 4096, 6, 8
D_FF = int(D * 8 / 3)
SEQ = 1024

# 1. 模型实例化
torch.manual_seed(42)
nx = NexusGPT(V, D, L, H, D_FF, SEQ).to('cuda')
bl = BaselineGPT(V, D, L, H, D_FF, SEQ).to('cuda')

# 检查 TTT LoRA rank
block = nx.blocks[0]
if hasattr(block, 'ttt'):
    r = block.ttt.ttt_rank
    has_wa = hasattr(block.ttt, 'W_A')
    print(f'\n  TTT LoRA rank={r}, has W_A={has_wa}')
    if has_wa:
        print(f'  W_A: {block.ttt.W_A.weight.shape}')
        print(f'  W_B: {block.ttt.W_B.weight.shape}')
    ttt_params = sum(p.numel() for p in block.ttt.parameters())
    print(f'  TTT params/layer: {ttt_params:,}')

# 2. 参数量对比
nx_params = sum(p.numel() for p in nx.parameters())
bl_params = sum(p.numel() for p in bl.parameters())
print(f'\n  Baseline: {bl_params:,} params')
print(f'  NEXUS:    {nx_params:,} params')
print(f'  差异:     {(nx_params-bl_params)/bl_params*100:+.1f}%')

# 3. 梯度健康
x = torch.randint(0, V, (2, SEQ), device='cuda')
y = torch.randint(0, V, (2, SEQ), device='cuda')

_, loss = nx(x, y)
loss.backward()
grads = [p.grad.norm().item() for p in nx.parameters() if p.grad is not None]
has_nan = any(math.isnan(g) for g in grads)
print(f'\n  Loss: {loss.item():.4f}')
print(f'  梯度: mean={np.mean(grads):.6f}, max={np.max(grads):.6f}, NaN={has_nan}')

# 4. TTT 在线学习
nx.eval()
with torch.no_grad():
    logits, _ = nx(x[:1])
    losses_seg = []
    for i in range(4):
        s, e = i*256, (i+1)*256
        seg_l = logits[0, s:e-1].reshape(-1, V)
        seg_t = x[0, s+1:e].reshape(-1)
        losses_seg.append(F.cross_entropy(seg_l, seg_t).item())
    slope = np.polyfit(range(4), losses_seg, 1)[0]

ok = "OK" if slope < 0 else "weak"
print(f'  TTT online learning: slope={slope:+.6f} [{ok}]')

# 5. 显存
torch.cuda.reset_peak_memory_stats()
nx.train(); nx.zero_grad()
_, loss2 = nx(x, y)
loss2.backward()
mem = torch.cuda.max_memory_allocated() / 1e6
print(f'  训练显存: {mem:.0f} MB')

total = time.time() - t0
print(f'\n  验证完成! 耗时: {total:.0f}s')
