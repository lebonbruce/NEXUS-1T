"""
NEXUS Scale-Up 深水区检测 — 5 项快速验证

在向 50M-1B 规模迈进前，用快速测试提前发现隐患：
  1. MLA 绝对维度 vs 压缩比 (d=1024: 2x vs 4x)
  2. TTT 显存爆炸测试 (D² 问题)
  3. DiffAttn λ 饱和度检查
  4. RoPE 长文本外推极限
  5. BF16 数值漂移检测
"""

import os
import sys
import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain"))
from models import BaselineGPT, NexusGPT, TTTLinear, DiffAttnMLA

DEVICE = "cuda"


# ============================================================
# TEST 1: MLA 绝对维度 vs 压缩比
# ============================================================

def test_mla_ratio():
    """
    在 d_model=1024 下对比 MLA 2x（瓶颈512维）vs 4x（瓶颈256维）。
    通过 monkeypatch get_mla_compression 来强制不同压缩比。
    """
    print("=" * 70)
    print("  TEST 1: MLA 绝对维度 vs 压缩比 (d=1024)")
    print("=" * 70)

    import models as m
    original_func = m.get_mla_compression

    V = 1024  # 小词表加速
    D = 1024
    L = 2    # 2 层足够看梯度
    H = 16
    SEQ = 512
    DFF = int(D * 4 * 2 / 3)

    configs = [
        ("MLA OFF (独立KV)", 1),
        ("MLA 2x (瓶颈=512维)", 2),
        ("MLA 4x (瓶颈=256维)", 4),
    ]

    print(f"\n  {'Config':<25s} | {'Grad Norm':>10s} | {'Init Loss':>10s} | {'Attn Grad':>10s} | {'Status'}")
    print(f"  {'-'*80}")

    for name, compression in configs:
        torch.manual_seed(42)
        torch.cuda.empty_cache()

        # Monkeypatch: 强制返回指定压缩比
        m.get_mla_compression = lambda d, c=compression: c

        attn = DiffAttnMLA(D, H, SEQ).to(DEVICE)
        tok_emb = nn.Embedding(V, D).to(DEVICE)
        head = nn.Linear(D, V, bias=False).to(DEVICE)
        ln = nn.LayerNorm(D).to(DEVICE)

        x_ids = torch.randint(0, V, (2, SEQ), device=DEVICE)
        y_ids = torch.randint(0, V, (2, SEQ), device=DEVICE)

        x = tok_emb(x_ids)
        out = attn(ln(x))
        logits = head(out)
        loss = F.cross_entropy(logits.view(-1, V), y_ids.view(-1))

        loss.backward()

        attn_grads = []
        for p in attn.parameters():
            if p.grad is not None:
                attn_grads.append(p.grad.norm().item())
        avg_attn_grad = np.mean(attn_grads) if attn_grads else 0

        all_grads = []
        for module in [tok_emb, attn, head, ln]:
            for p in module.parameters():
                if p.grad is not None:
                    all_grads.append(p.grad.norm().item())
        avg_grad = np.mean(all_grads)

        has_nan = any(math.isnan(g) for g in all_grads)
        has_vanish = any(g < 1e-7 for g in attn_grads)

        if has_nan:
            status = "❌ NaN!"
        elif has_vanish:
            status = "⚠️ 梯度消失"
        elif avg_attn_grad < 0.01:
            status = "⚠️ 梯度弱"
        else:
            status = "✅ 健康"

        print(f"  {name:<25s} | {avg_grad:>10.6f} | {loss.item():>10.4f} | {avg_attn_grad:>10.6f} | {status}")

        del attn, tok_emb, head, ln
        torch.cuda.empty_cache()

    # 恢复原始函数
    m.get_mla_compression = original_func

    print(f"\n  结论: 如果 4x 的梯度明显弱于 2x，说明在 d=1024 下仍需谨慎使用高压缩比。")


# ============================================================
# TEST 2: TTT 显存爆炸测试 (D² 问题)
# ============================================================

def test_ttt_memory():
    """
    测试 TTT-Linear 在不同 d_model 下的显存占用。
    关键：W_all tensor 的大小 = [B, n_batches, D, D]
    """
    print("\n" + "=" * 70)
    print("  TEST 2: TTT 显存爆炸测试 (D² 问题)")
    print("=" * 70)

    configs = [
        # (d_model, seq_len, batch, label)
        (256,  1024, 4, "Small (256)"),
        (512,  1024, 4, "Medium (512)"),
        (768,  1024, 2, "Large (768)"),
        (1024, 1024, 2, "XL (1024)"),
        (1024, 2048, 1, "XL+Long (1024x2048)"),
        (2048, 1024, 1, "XXL (2048)"),
        (2048, 2048, 1, "XXL+Long (2048x2048)"),
    ]

    print(f"\n  {'Config':<25s} | {'W_all Size':>15s} | {'D² MB':>8s} | {'Peak MB':>10s} | {'Status'}")
    print(f"  {'-'*80}")

    for d, seq, batch, label in configs:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        # 理论计算 W_all 大小
        mini_bs = 16
        n_batches = seq // mini_bs
        w_all_elements = batch * n_batches * d * d
        w_all_mb = w_all_elements * 4 / 1024**2  # float32

        oom = False
        peak = 0
        status = ""

        ttt = TTTLinear(d, mini_batch_size=mini_bs).to(DEVICE)
        x = torch.randn(batch, seq, d, device=DEVICE)

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**2

        with torch.no_grad():
            try:
                out = ttt(x)
                peak = torch.cuda.max_memory_allocated() / 1024**2
                fwd_mem = peak - mem_before
                status = "✅ OK"
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom = True
                    peak = -1
                    fwd_mem = -1
                    status = "💥 OOM!"
                else:
                    raise

        if oom:
            print(f"  {label:<25s} | {w_all_mb:>12.0f} MB | {d*d*4/1024**2:>7.1f} | {'OOM':>10s} | {status}")
        else:
            print(f"  {label:<25s} | {w_all_mb:>12.0f} MB | {d*d*4/1024**2:>7.1f} | {fwd_mem:>9.0f}M | {status}")

        del ttt, x
        if not oom:
            del out
        torch.cuda.empty_cache()

        if oom:
            # 计算需要的 LoRA rank 来解决
            max_vram = 7500  # RTX 4060 ~7.5GB 可用
            # W_all_mb ≈ batch * n_batches * D * D * 4 / 1024²
            # LoRA: batch * n_batches * D * r * 2 * 4 / 1024²
            # 需要 r 使得: batch * n_batches * D * r * 2 * 4 / 1024² < max_vram * 0.5
            target_mb = max_vram * 0.3  # 留 30% 给 TTT
            r = int(target_mb * 1024**2 / (batch * n_batches * d * 2 * 4))
            print(f"    → LoRA-TTT 建议: rank={r} 可将显存降至 ~{target_mb:.0f}MB")

    print(f"\n  GPU: RTX 4060 (8GB VRAM ≈ 7.5GB 可用)")
    print(f"  结论: D² 增长意味着 d_model=2048 时 TTT 一定 OOM。")
    print(f"  如果要 Scale-up 到 1B+，必须实现 LoRA-style TTT。")


# ============================================================
# TEST 3: DiffAttn λ 分布检查
# ============================================================

def test_lambda_distribution():
    """
    检查 DiffAttn 中 λ 参数的初始化分布和每层差异。
    健康状态：不同层的 λ 应该有差异（浅层多滤噪，深层保语义）。
    """
    print("\n" + "=" * 70)
    print("  TEST 3: DiffAttn λ 参数分布检查")
    print("=" * 70)

    V = 1024
    D = 512
    L = 6
    H = 8
    SEQ = 512
    DFF = int(D * 4 * 2 / 3)

    torch.manual_seed(42)
    model = NexusGPT(V, D, L, H, DFF, SEQ).to(DEVICE)

    # 收集每层的 λ 参数
    print(f"\n  {'Layer':>8s} | {'λ_q1 mean':>10s} | {'λ_k1 mean':>10s} | {'λ_init':>10s} | {'exp(λ)':>8s}")
    print(f"  {'-'*55}")

    lambda_values = []
    for i, block in enumerate(model.blocks):
        attn = block.attn
        # DiffAttn 的 lambda 参数
        lq1 = attn.lambda_q1.data.mean().item()
        lk1 = attn.lambda_k1.data.mean().item()
        lq2 = attn.lambda_q2.data.mean().item()
        lk2 = attn.lambda_k2.data.mean().item()

        # λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
        lambda_val = math.exp(lq1 * lk1) - math.exp(lq2 * lk2) + attn.lambda_init
        lambda_values.append(lambda_val)

        print(f"  {i:>8d} | {lq1:>10.4f} | {lk1:>10.4f} | {attn.lambda_init:>10.4f} | {lambda_val:>8.4f}")

    # 分析
    mean_lambda = np.mean(lambda_values)
    std_lambda = np.std(lambda_values)

    print(f"\n  λ 统计: mean={mean_lambda:.4f}, std={std_lambda:.6f}")

    if std_lambda < 1e-6:
        print(f"  ⚠️ 所有层 λ 完全相同（初始化导致）")
        print(f"     训练后应观察是否出现层间分化")
    elif std_lambda > 0.1:
        print(f"  ✅ λ 在层间有显著差异，差分机制正常")
    else:
        print(f"  ℹ️ λ 初始化时差异微小，这是正常的（训练后会分化）")

    if 0.1 < mean_lambda < 0.9:
        print(f"  ✅ λ 均值在合理范围 (0.1-0.9)，差分注意力有效")
    elif mean_lambda >= 0.9:
        print(f"  ⚠️ λ 接近 1.0 → 差分几乎全减去了 attn2，退化风险")
    elif mean_lambda <= 0.1:
        print(f"  ⚠️ λ 接近 0.0 → 差分几乎无效，退化为标准注意力")

    del model
    torch.cuda.empty_cache()


# ============================================================
# TEST 4: RoPE 长文本外推极限
# ============================================================

def test_rope_extrapolation():
    """
    测试 RoPE 在超出训练长度时的数值稳定性。
    不需要预训练：只看注意力分数的分布是否合理。
    """
    print("\n" + "=" * 70)
    print("  TEST 4: RoPE 长文本外推极限")
    print("=" * 70)

    V = 1024
    D = 256  # 小模型测试
    L = 2
    H = 8
    DFF = 1024
    BATCH = 1

    test_lengths = [512, 1024, 2048, 4096, 8192]

    print(f"\n  {'Seq Len':>8s} | {'BL Loss':>10s} | {'NX Loss':>10s} | {'BL LogitStd':>12s} | {'NX LogitStd':>12s} | {'Status'}")
    print(f"  {'-'*75}")

    for seq in test_lengths:
        torch.cuda.empty_cache()
        gc.collect()

        bl_ok, nx_ok = True, True
        bl_loss, nx_loss = 0, 0
        bl_std, nx_std = 0, 0

        for name, model_cls, dff in [
            ("BL", BaselineGPT, DFF),
            ("NX", NexusGPT, int(DFF * 2 / 3)),
        ]:
            torch.manual_seed(42)
            oom = False

            model = model_cls(V, D, L, H, dff, seq).to(DEVICE).eval()
            x = torch.randint(0, V, (BATCH, seq), device=DEVICE)
            y = torch.randint(0, V, (BATCH, seq), device=DEVICE)

            with torch.no_grad():
                try:
                    logits, loss = model(x, y)
                    l = loss.item()
                    s = logits.std().item()
                    has_nan = math.isnan(l) or math.isnan(s) or math.isinf(l)
                except RuntimeError:
                    oom = True
                    l, s = -1, -1
                    has_nan = True

            if name == "BL":
                bl_loss, bl_std = l, s
                bl_ok = not has_nan and not oom
            else:
                nx_loss, nx_std = l, s
                nx_ok = not has_nan and not oom

            del model, x, y
            torch.cuda.empty_cache()

        # 状态判断
        if not bl_ok or not nx_ok:
            status = "💥 NaN/OOM"
        elif bl_std > 100 or nx_std > 100:
            status = "⚠️ Logit 爆炸"
        else:
            status = "✅ 稳定"

        bl_loss_s = f"{bl_loss:.4f}" if bl_ok else "OOM"
        nx_loss_s = f"{nx_loss:.4f}" if nx_ok else "OOM"
        bl_std_s = f"{bl_std:.4f}" if bl_ok else "N/A"
        nx_std_s = f"{nx_std:.4f}" if nx_ok else "N/A"

        print(f"  {seq:>8d} | {bl_loss_s:>10s} | {nx_loss_s:>10s} | {bl_std_s:>12s} | {nx_std_s:>12s} | {status}")

    print(f"\n  RoPE base=10000 (标准)")
    print(f"  如果超长序列 logit std 暴增或出现 NaN：")
    print(f"    → 需要 YaRN 或 Dynamic NTK scaling")
    print(f"    → 或提高 RoPE base 到 100000+")


# ============================================================
# TEST 5: BF16 数值漂移检测
# ============================================================

def test_bf16_drift():
    """
    对比 FP32 vs BF16 下 TTT cumsum 的数值差异。
    如果差异大于阈值，说明需要对 TTT 内部的累加强制 FP32。
    """
    print("\n" + "=" * 70)
    print("  TEST 5: BF16 数值漂移检测 (TTT cumsum)")
    print("=" * 70)

    D = 512
    SEQ = 1024
    BATCH = 2

    torch.manual_seed(42)

    # FP32 参考
    ttt_f32 = TTTLinear(D, mini_batch_size=16).to(DEVICE).float()
    x_f32 = torch.randn(BATCH, SEQ, D, device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        out_f32 = ttt_f32(x_f32)

    # BF16
    ttt_bf16 = TTTLinear(D, mini_batch_size=16).to(DEVICE).bfloat16()
    # 复制相同的权重
    ttt_bf16.load_state_dict(ttt_f32.state_dict())
    ttt_bf16 = ttt_bf16.bfloat16()
    x_bf16 = x_f32.bfloat16()

    with torch.no_grad():
        out_bf16 = ttt_bf16(x_bf16).float()

    # FP16
    has_fp16 = True
    try:
        ttt_f16 = TTTLinear(D, mini_batch_size=16).to(DEVICE).half()
        ttt_f16.load_state_dict(ttt_f32.state_dict())
        ttt_f16 = ttt_f16.half()
        x_f16 = x_f32.half()

        with torch.no_grad():
            out_f16 = ttt_f16(x_f16).float()
    except RuntimeError:
        has_fp16 = False

    # 对比
    # 相对误差
    bf16_err = (out_bf16 - out_f32).abs() / (out_f32.abs() + 1e-8)
    bf16_rel_err = bf16_err.mean().item()
    bf16_max_err = bf16_err.max().item()
    bf16_nan = torch.isnan(out_bf16).any().item()

    print(f"\n  FP32 output: mean={out_f32.mean().item():.6f}, std={out_f32.std().item():.6f}")
    print(f"\n  {'Precision':<10s} | {'Mean Rel Err':>12s} | {'Max Rel Err':>12s} | {'NaN?':>6s} | {'Status'}")
    print(f"  {'-'*60}")

    if bf16_nan:
        status = "💥 NaN!"
    elif bf16_rel_err > 0.1:
        status = "❌ 严重漂移"
    elif bf16_rel_err > 0.01:
        status = "⚠️ 轻微漂移"
    else:
        status = "✅ 安全"
    print(f"  {'BF16':<10s} | {bf16_rel_err:>12.6f} | {bf16_max_err:>12.6f} | {'Yes' if bf16_nan else 'No':>6s} | {status}")

    if has_fp16:
        f16_err = (out_f16 - out_f32).abs() / (out_f32.abs() + 1e-8)
        f16_rel_err = f16_err.mean().item()
        f16_max_err = f16_err.max().item()
        f16_nan = torch.isnan(out_f16).any().item()

        if f16_nan:
            status = "💥 NaN!"
        elif f16_rel_err > 0.1:
            status = "❌ 严重漂移"
        elif f16_rel_err > 0.01:
            status = "⚠️ 轻微漂移"
        else:
            status = "✅ 安全"
        print(f"  {'FP16':<10s} | {f16_rel_err:>12.6f} | {f16_max_err:>12.6f} | {'Yes' if f16_nan else 'No':>6s} | {status}")

    # 建议
    print(f"\n  结论:")
    if bf16_nan:
        print(f"  ❌ BF16 导致 NaN！TTT 的 cumsum 必须强制 FP32")
    elif bf16_rel_err > 0.01:
        print(f"  ⚠️ BF16 有可测量的漂移。建议 TTT 内部 cumsum 使用 FP32")
        print(f"     方法: 在 TTTLinear.forward 中: cum_grad = cum_grad.float() 再转回")
    else:
        print(f"  ✅ BF16 数值稳定，可以安全使用混合精度训练")


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("  NEXUS Scale-Up 深水区检测")
    print(f"  Device: {DEVICE}")
    print(f"  目标: 在 Scale-up 前发现所有隐患")
    print("=" * 70)

    t0 = time.time()

    test_mla_ratio()           # ~3s
    test_ttt_memory()          # ~5s
    test_lambda_distribution() # ~2s
    test_rope_extrapolation()  # ~5s
    test_bf16_drift()          # ~3s

    total = time.time() - t0
    print(f"\n\n{'=' * 70}")
    print(f"  深水区检测完成！总耗时: {total:.0f}s")
    print(f"{'=' * 70}")

    print("""
  === Scale-Up 行动清单 ===
  根据以上结果，在向 50M+ 迈进前需要：
  
  🔴 阻塞项（必须解决才能 scale-up）：
     - TTT OOM → 实现 LoRA-TTT（低秩更新）
     - BF16 NaN → TTT cumsum 强制 FP32
  
  🟡 注意项（可以先 scale-up 再观察）：
     - MLA 压缩比 → 根据梯度健康度动态选择
     - λ 饱和 → 训练时监控每层 λ 分布
     - RoPE 外推 → 如果 PPL 爆炸则切换 YaRN
    """)


if __name__ == "__main__":
    main()
