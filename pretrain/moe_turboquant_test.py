"""
MoE + TurboQuant — 7 秒快速验证

测试内容：
  1. MoE 梯度健康 + expert 负载均衡
  2. Dense vs MoE 显存对比
  3. MoE 对学习速率的影响
  4. TurboQuant 金字塔快照压缩 vs 精度
  5. 更新后的 1T 推理配置
"""

import os, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain"))
from models import NexusGPT, SwiGLUFFN, MoESwiGLUFFN, TTTLinear

DEVICE = "cuda"


# ============================================================
# TurboQuant: 快照量化压缩
# ============================================================

class TurboQuant:
    """
    TurboQuant 金字塔快照压缩。

    原理：
      1. Fingerprint (D 维向量) → INT8 量化 (1/4 大小)
      2. W_grad (D×D 矩阵) → SVD 低秩 + INT8 量化
      3. Routing 精度检测：量化前后的 cosine similarity

    目标：95%+ 信息保留，75%+ 存储节省
    """

    @staticmethod
    def quantize_fp(tensor):
        """FP32/BF16 → INT8 对称量化。"""
        scale = tensor.abs().max() / 127.0
        if scale == 0:
            return torch.zeros_like(tensor, dtype=torch.int8), scale
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    @staticmethod
    def dequantize_fp(quantized, scale):
        """INT8 → FP32 反量化。"""
        return quantized.float() * scale

    @staticmethod
    def compress_snapshot(fingerprint, w_grad, rank=8):
        """
        压缩一个金字塔快照。

        输入：
          fingerprint: [D] 向量
          w_grad: [D, D] 矩阵

        输出：
          compressed: dict, 包含量化后的数据
          原始大小 vs 压缩后大小
        """
        D = fingerprint.size(0)
        orig_bytes = (D + D * D) * 4  # FP32

        # 1. Fingerprint → INT8
        fp_q, fp_scale = TurboQuant.quantize_fp(fingerprint)

        # 2. W_grad → SVD 低秩 + INT8
        U, S, Vh = torch.linalg.svd(w_grad, full_matrices=False)
        U_r = U[:, :rank]        # [D, rank]
        S_r = S[:rank]            # [rank]
        Vh_r = Vh[:rank, :]       # [rank, D]

        # 对 U_r 和 Vh_r 做 INT8 量化
        U_q, U_scale = TurboQuant.quantize_fp(U_r)
        Vh_q, Vh_scale = TurboQuant.quantize_fp(Vh_r)
        # S_r 保持 FP16（只有 rank 个值，很小）
        S_fp16 = S_r.half()

        # 压缩后大小
        comp_bytes = (
            D * 1 + 4              # fp_q (INT8) + fp_scale (FP32)
            + D * rank * 1 + 4     # U_q (INT8) + U_scale
            + rank * 2             # S_fp16 (FP16)
            + rank * D * 1 + 4     # Vh_q (INT8) + Vh_scale
        )

        return {
            "fp_q": fp_q, "fp_scale": fp_scale,
            "U_q": U_q, "U_scale": U_scale,
            "S_fp16": S_fp16,
            "Vh_q": Vh_q, "Vh_scale": Vh_scale,
            "rank": rank,
        }, orig_bytes, comp_bytes

    @staticmethod
    def decompress_snapshot(compressed):
        """反量化 + 低秩重建。"""
        fp = TurboQuant.dequantize_fp(compressed["fp_q"], compressed["fp_scale"])

        U = TurboQuant.dequantize_fp(compressed["U_q"], compressed["U_scale"])
        S = compressed["S_fp16"].float()
        Vh = TurboQuant.dequantize_fp(compressed["Vh_q"], compressed["Vh_scale"])

        w_grad = U @ torch.diag(S) @ Vh
        return fp, w_grad


def test_moe():
    """测试 MoE 梯度健康 + 显存 + expert 分布。"""
    print("=" * 70)
    print("  TEST 1: MoE 梯度健康 + Expert 负载均衡")
    print("=" * 70)

    D = 512
    D_FF = int(D * 8 / 3)
    SEQ = 256
    BATCH = 2
    V = 1024

    # Dense
    torch.manual_seed(42)
    dense_ffn = SwiGLUFFN(D, D_FF).to(DEVICE)
    dense_params = sum(p.numel() for p in dense_ffn.parameters())

    # MoE
    torch.manual_seed(42)
    moe_ffn = MoESwiGLUFFN(D, D_FF, n_experts=8, top_k=2).to(DEVICE)
    moe_params = sum(p.numel() for p in moe_ffn.parameters())

    x = torch.randn(BATCH, SEQ, D, device=DEVICE)

    # Dense forward + memory
    torch.cuda.reset_peak_memory_stats()
    dense_out = dense_ffn(x)
    dense_mem = torch.cuda.max_memory_allocated() / 1e6

    # MoE forward + memory
    torch.cuda.reset_peak_memory_stats()
    moe_ffn.train()
    moe_out = moe_ffn(x)
    moe_mem = torch.cuda.max_memory_allocated() / 1e6

    # 梯度检查
    loss_moe = moe_out.sum() + moe_ffn.aux_loss
    loss_moe.backward()

    grad_norms = {}
    has_nan = False
    for name, p in moe_ffn.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            grad_norms[name] = gn
            if math.isnan(gn):
                has_nan = True

    # Expert 分布（用新的 forward 检查）
    moe_ffn.eval()
    with torch.no_grad():
        router_logits = moe_ffn.router(x.view(-1, D))
        router_probs = F.softmax(router_logits, dim=-1)
        expert_load = router_probs.mean(dim=0)

    print(f"""
  Dense FFN:
    参数: {dense_params:,}
    显存: {dense_mem:.0f} MB
    激活参数/token: {dense_params:,} (100%)

  MoE FFN (8 experts, top-2):
    参数: {moe_params:,}
    显存: {moe_mem:.0f} MB
    激活参数/token: {dense_params * 2:,} ({2/8*100:.0f}% of total)
    
  参数效率: {moe_params/dense_params:.1f}x 总参数, {2/8:.0%} 计算量
  梯度健康: {'✅ 无 NaN' if not has_nan else '❌ 有 NaN'}
  Aux Loss: {moe_ffn.aux_loss:.6f}
  
  Expert 负载分布 (理想=0.125 each):""")
    for i, load in enumerate(expert_load):
        bar = "█" * int(load.item() * 80)
        ideal_diff = abs(load.item() - 1.0 / 8) / (1.0 / 8) * 100
        print(f"    Expert {i}: {load.item():.4f} {bar} (偏差 {ideal_diff:.0f}%)")

    max_load = expert_load.max().item()
    min_load = expert_load.min().item()
    balance = min_load / max_load
    print(f"\n  负载均衡度: {balance:.4f} (1.0=完美, <0.5=崩塌)")
    print(f"  状态: {'✅ 均衡' if balance > 0.3 else '⚠️ 不均衡'}")

    del dense_ffn, moe_ffn
    torch.cuda.empty_cache()


def test_moe_model():
    """测试完整 NEXUS 模型（MoE 启用）的梯度健康。"""
    print(f"\n{'='*70}")
    print(f"  TEST 2: NexusGPT + MoE 完整模型验证")
    print(f"{'='*70}")

    D = 1024
    V = 2048
    L = 2
    H = 16
    D_FF = int(D * 8 / 3)
    D_FF = ((D_FF + 255) // 256) * 256
    SEQ = 1024

    torch.manual_seed(42)
    model = NexusGPT(V, D, L, H, D_FF, SEQ).to(DEVICE)

    # 检查 MoE 自动启用
    has_moe = any(hasattr(b, 'use_moe') and b.use_moe for b in model.blocks)
    print(f"\n  MoE 自动启用: {has_moe}")

    x = torch.randint(0, V, (1, SEQ), device=DEVICE)
    y = torch.randint(0, V, (1, SEQ), device=DEVICE)

    torch.cuda.reset_peak_memory_stats()
    logits, loss = model(x, y)

    # 收集 aux loss
    total_aux = 0
    for block in model.blocks:
        if hasattr(block, 'ffn') and hasattr(block.ffn, 'aux_loss'):
            total_aux += block.ffn.aux_loss

    total_loss = loss + total_aux
    total_loss.backward()

    mem = torch.cuda.max_memory_allocated() / 1e6

    # 梯度统计
    all_grads = []
    for p in model.parameters():
        if p.grad is not None:
            all_grads.append(p.grad.norm().item())

    print(f"  CE Loss: {loss.item():.4f}")
    print(f"  Aux Loss: {total_aux:.6f}")
    print(f"  显存: {mem:.0f} MB")
    print(f"  梯度: mean={np.mean(all_grads):.6f}, max={np.max(all_grads):.6f}")
    print(f"  NaN: {any(math.isnan(g) for g in all_grads)}")
    print(f"  状态: {'✅ 健康' if not any(math.isnan(g) for g in all_grads) else '❌ NaN'}")

    del model
    torch.cuda.empty_cache()


def test_turboquant():
    """测试 TurboQuant 对金字塔快照的压缩效果。"""
    print(f"\n{'='*70}")
    print(f"  TEST 3: TurboQuant 金字塔快照压缩")
    print(f"{'='*70}")

    D = 256

    # 生成模拟快照
    torch.manual_seed(42)
    fingerprint = torch.randn(D, device=DEVICE)
    w_grad = torch.randn(D, D, device=DEVICE) * 0.1

    ranks = [4, 8, 16, 32, 64, 128]

    print(f"\n  {'Rank':>6s} | {'压缩比':>8s} | {'FP cos':>8s} | {'W cos':>8s} | {'W MSE':>12s} | {'状态'}")
    print(f"  {'-'*65}")

    for rank in ranks:
        comp, orig_bytes, comp_bytes = TurboQuant.compress_snapshot(
            fingerprint, w_grad, rank=rank
        )

        fp_rec, w_rec = TurboQuant.decompress_snapshot(comp)

        # 指纹精度
        fp_cos = F.cosine_similarity(fingerprint.unsqueeze(0), fp_rec.unsqueeze(0)).item()

        # W_grad 精度
        w_cos = F.cosine_similarity(
            w_grad.flatten().unsqueeze(0),
            w_rec.flatten().unsqueeze(0)
        ).item()
        w_mse = ((w_grad - w_rec) ** 2).mean().item()

        ratio = orig_bytes / comp_bytes

        status = "⭐" if fp_cos > 0.999 and w_cos > 0.95 else "✅" if w_cos > 0.9 else "⚠️"

        print(f"  {rank:>6d} | {ratio:>7.1f}x | {fp_cos:>+8.6f} | {w_cos:>+8.4f} | {w_mse:>12.8f} | {status}")

    # 用 1T 的 D=21504 计算实际存储节省
    print(f"\n  === 1T 模型 (D=21504) 100M token 实际存储 ===")
    D_1t = 21504
    n_snapshots = 100_000_000 // 1024  # 粒度=1K

    for rank in [8, 16, 32]:
        # 每快照压缩后大小
        per_snap = D_1t * 1 + 4 + D_1t * rank * 1 + 4 + rank * 2 + rank * D_1t * 1 + 4
        total = n_snapshots * per_snap

        # 原始大小
        per_snap_orig = (D_1t + 2 * D_1t * rank) * 4  # SVD low-rank FP32
        total_orig = n_snapshots * per_snap_orig

        def fmt(b):
            if b >= 1e12: return f"{b/1e12:.1f}TB"
            elif b >= 1e9: return f"{b/1e9:.1f}GB"
            elif b >= 1e6: return f"{b/1e6:.0f}MB"
            else: return f"{b/1e3:.0f}KB"

        print(f"    rank={rank:>3d}: {fmt(total_orig)} → {fmt(total)} "
              f"(节省 {(1-total/total_orig)*100:.0f}%)")


def test_updated_config():
    """更新后的 1T MoE 推理配置。"""
    print(f"\n{'='*70}")
    print(f"  TEST 4: NEXUS-1T-MoE 最终推理配置")
    print(f"{'='*70}")

    D = 21504
    L = 152
    H = 168
    V = 128_000
    n_experts = 8
    top_k = 2

    # 参数计算（MoE 版）
    # DiffAttnMLA 同前
    attn = 2 * D * D + 3 * D * D // 4  # q_proj + kv down/up
    # TTT (LoRA, rank=D//4)
    r = D // 4
    ttt = D * D + 2 * D * r + 2 * (D * D + D)  # theta + LoRA + gates
    # MoE SwiGLU: n_experts * 3 * D * d_ff + router
    d_ff = ((int(D * 8 / 3) + 255) // 256) * 256
    ffn = n_experts * 3 * D * d_ff + D * n_experts

    per_layer = attn + ttt + ffn
    total = L * per_layer + 2 * V * D

    # 激活参数（每 token 只用 top-2）
    ffn_active = top_k * 3 * D * d_ff
    active_per_layer = attn + ttt + ffn_active
    active_total = L * active_per_layer

    def fmt(b):
        if b >= 1e12: return f"{b/1e12:.1f}T"
        elif b >= 1e9: return f"{b/1e9:.1f}B"
        elif b >= 1e6: return f"{b/1e6:.0f}M"
        else: return f"{b/1e3:.0f}K"

    def fmt_mem(b):
        if b >= 1e12: return f"{b/1e12:.1f} TB"
        elif b >= 1e9: return f"{b/1e9:.1f} GB"
        elif b >= 1e6: return f"{b/1e6:.0f} MB"
        else: return f"{b/1e3:.0f} KB"

    model_mem_bf16 = total * 2
    model_mem_int4 = total * 0.5  # INT4 = 0.5 bytes/param

    # 推理显存计算
    window = 8192
    kv = 2 * window * D * 2 // 4 * L  # MLA 4x
    ttt_w = L * 2 * D * r * 2  # LoRA
    # 金字塔（TurboQuant, rank=8, INT8）
    snap_rank = 8
    n_snaps = 100_000_000 // 1024
    per_snap = D * 1 + 2 * D * snap_rank * 1 + snap_rank * 2  # INT8
    pyramid = n_snaps * per_snap

    print(f"""
  ╔═══════════════════════════════════════════════════════╗
  ║  NEXUS-1T-MoE 最终架构                                ║
  ╠═══════════════════════════════════════════════════════╣
  ║  d_model      = {D:>6,d}                              ║
  ║  n_layers     = {L:>6d}                                ║
  ║  n_heads      = {H:>6d}                                ║
  ║  d_ff         = {d_ff:>6,d}    (SwiGLU)                 ║
  ║  n_experts    = {n_experts:>6d}      (MoE)                   ║
  ║  top_k        = {top_k:>6d}      (每 token 激活)            ║
  ║  ttt_rank     = {r:>6,d}    (LoRA D//4)                ║
  ╠═══════════════════════════════════════════════════════╣
  ║  总参数        = {fmt(total):>6s}                              ║
  ║  激活参数/token = {fmt(active_total):>6s}  ({active_total/total*100:.0f}%)            ║
  ║  推理速度      ≈ {fmt(active_total)} dense 模型               ║
  ╚═══════════════════════════════════════════════════════╝

  === 推理显存 (100M tokens) ===
  
                         BF16          INT4+TurboQuant
  模型参数:          {fmt_mem(model_mem_bf16):>10s}      {fmt_mem(model_mem_int4):>10s}
  KV Cache:          {fmt_mem(kv):>10s}      {fmt_mem(kv//2):>10s}
  TTT W (LoRA):      {fmt_mem(ttt_w):>10s}      {fmt_mem(ttt_w//4):>10s}
  金字塔 (TurboQ):   {fmt_mem(pyramid):>10s}      {fmt_mem(pyramid):>10s}
  ─────────────────────────────────────────
  总计:              {fmt_mem(model_mem_bf16 + kv + ttt_w + pyramid):>10s}      {fmt_mem(model_mem_int4 + kv//2 + ttt_w//4 + pyramid):>10s}
  GPU (A100 80G):    {math.ceil((model_mem_bf16+kv+ttt_w+pyramid)/(80*1e9)):>3d} 张           {math.ceil((model_mem_int4+kv//2+ttt_w//4+pyramid)/(80*1e9)):>3d} 张
  GPU (H200 141G):   {math.ceil((model_mem_bf16+kv+ttt_w+pyramid)/(141*1e9)):>3d} 张           {math.ceil((model_mem_int4+kv//2+ttt_w//4+pyramid)/(141*1e9)):>3d} 张

  === 训练成本 ===
  训练显存: ~{fmt_mem(total * 2 + total * 8 + total * 2)} (模型+优化器+梯度)
  训练 GPU: ~{math.ceil((total*12)/(80*1e9))} × A100 80GB, 90 天
  估计成本: ~${math.ceil((total*12)/(80*1e9)) * 2 * 720 * 90 / 1000:.0f}K

  === 推理速度 vs Dense ===
  Dense 1T:  每 token 计算 {fmt(total)} 参数
  MoE 1T:    每 token 计算 {fmt(active_total)} 参数 ({active_total/total:.0%})
  加速比:    ~{total/active_total:.1f}x
    """)


def main():
    t0 = time.time()

    test_moe()
    test_moe_model()
    test_turboquant()
    test_updated_config()

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  全部完成！总耗时: {total:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
