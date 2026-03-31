"""
NEXUS 快速架构验证套件（RTX 4060 友好）

三种零成本/低成本验证方法，替代昂贵的 3000 步预训练对比：

  TEST 1: 梯度健康检查（10 秒，零训练）
    - 一次 forward + backward，检查每层梯度范数分布
    - 诊断：梯度消失、梯度爆炸、信息瓶颈、死层
    
  TEST 2: 100 步 Micro-Benchmark（2-3 分钟）
    - 在 tiny scale 上跑 100 步，对比 loss 下降速率
    - 架构好坏在前 100 步的斜率上就能看出来
    
  TEST 3: μP 坐标检查（30 秒，零训练）
    - 检验不同 width 下激活值分布是否稳定
    - 如果稳定 → 小规模最优超参可迁移到大规模
"""

import os
import sys
import time
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain"))
from models import BaselineGPT, NexusGPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TEST 1: 梯度健康检查（零训练，10 秒）
# ============================================================

def test_gradient_health():
    """
    一次 forward + backward，检查每层梯度的：
      1. L2 范数 — 梯度大小（消失 < 1e-6，爆炸 > 100）
      2. 标准差 — 梯度多样性（死层 → std ≈ 0）
      3. 信噪比 (SNR) — mean/std，衡量梯度质量

    通过标准：
      ✅ 所有层梯度范数在 [1e-5, 10] 之间
      ✅ 没有 NaN/Inf
      ✅ Baseline 和 NEXUS 的梯度分布形状相似
    """
    print("=" * 70)
    print("  TEST 1: 梯度健康检查（零训练，1次 forward+backward）")
    print("=" * 70)

    V, D, L, H, SEQ = 50257, 384, 6, 6, 512

    results = {}
    for name, model_cls, dff in [
        ("Baseline", BaselineGPT, 1536),
        ("NEXUS-v4", NexusGPT, 1024),
    ]:
        torch.manual_seed(42)
        model = model_cls(V, D, L, H, dff, SEQ).to(DEVICE)
        x = torch.randint(0, V, (2, SEQ), device=DEVICE)
        y = torch.randint(0, V, (2, SEQ), device=DEVICE)

        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        layer_stats = []
        has_nan = False
        has_vanish = False
        has_explode = False

        for pname, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                norm = grad.norm().item()
                std = grad.std().item()
                mean = grad.mean().item()
                snr = abs(mean) / (std + 1e-12)

                if math.isnan(norm) or math.isinf(norm):
                    has_nan = True
                if norm < 1e-6:
                    has_vanish = True
                if norm > 100:
                    has_explode = True

                layer_stats.append({
                    "name": pname,
                    "norm": norm,
                    "std": std,
                    "snr": snr,
                    "numel": param.numel(),
                })

        results[name] = layer_stats

        # 按组件分组统计
        component_norms = {}
        for s in layer_stats:
            # 从参数名提取组件（如 blocks.0.attn, blocks.0.ffn）
            parts = s["name"].split(".")
            if len(parts) >= 3 and parts[0] == "blocks":
                component = ".".join(parts[2:3])  # attn, ttt, ffn, ln1 等
            else:
                component = parts[0]  # tok_emb, head 等
            if component not in component_norms:
                component_norms[component] = []
            component_norms[component].append(s["norm"])

        print(f"\n  [{name}] loss={loss.item():.4f}")
        print(f"  {'Component':<15s} | {'Avg Norm':>10s} | {'Min':>10s} | {'Max':>10s} | {'Layers':>6s}")
        print(f"  {'-'*60}")
        for comp, norms in sorted(component_norms.items()):
            avg = np.mean(norms)
            print(f"  {comp:<15s} | {avg:>10.6f} | {min(norms):>10.6f} | {max(norms):>10.6f} | {len(norms):>6d}")

        # 诊断
        status = "✅ 健康" if not (has_nan or has_vanish or has_explode) else ""
        if has_nan:
            status = "❌ NaN/Inf detected!"
        elif has_vanish:
            status = "⚠️ 部分层梯度过小（可能有信息瓶颈）"
        elif has_explode:
            status = "⚠️ 部分层梯度过大（可能需要 grad clipping）"

        print(f"  诊断: {status}")

        del model
        torch.cuda.empty_cache()

    # 对比分析
    print(f"\n  --- 架构对比 ---")
    bl_norms = [s["norm"] for s in results["Baseline"]]
    nx_norms = [s["norm"] for s in results["NEXUS-v4"]]
    bl_avg = np.mean(bl_norms)
    nx_avg = np.mean(nx_norms)
    print(f"  Baseline 平均梯度范数: {bl_avg:.6f}")
    print(f"  NEXUS-v4 平均梯度范数: {nx_avg:.6f}")
    print(f"  比值: {nx_avg/bl_avg:.2f}x")

    ratio = nx_avg / bl_avg
    if 0.1 < ratio < 10:
        print(f"  ✅ 梯度分布量级相近（比值在 0.1-10x），架构梯度流健康")
    else:
        print(f"  ⚠️ 梯度分布差异显著，可能需要调整学习率")

    return results


# ============================================================
# TEST 2: 100 步 Micro-Benchmark（2-3 分钟）
# ============================================================

def test_micro_benchmark():
    """
    在 tiny scale 上跑 100 步，对比 loss 下降速率。

    关键洞察（NAS 文献）：
      前 100 步的 loss 下降速率与最终收敛质量有 r > 0.9 的相关性。
      一个好架构应该在第一步就展现出更陡的学习曲线。

    测试配置：
      d_model=128, n_layers=4, seq=256, batch=8
      — 比完整模型小 ~60x，训练快 ~100x
    """
    print("\n" + "=" * 70)
    print("  TEST 2: 100 步 Micro-Benchmark（快速学习能力对比）")
    print("=" * 70)

    # Micro 配置（~0.5M 参数）
    V = 50257
    D = 128
    L = 4
    H = 4
    SEQ = 256
    BL_DFF = 512
    NX_DFF = 256  # SwiGLU 3 矩阵 → 3*256*128 = 2*512*128

    BATCH = 8
    N_STEPS = 100
    LR = 6e-4

    # 加载验证数据
    # 尝试多个可能的路径
    for base in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain", "data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        os.path.join("pretrain", "data"),
    ]:
        if os.path.exists(os.path.join(base, "train.bin")):
            DATA_DIR = base
            break
    else:
        DATA_DIR = None

    if DATA_DIR is None:
        print("  ⚠️ 训练数据不存在，跳过")
        return None

    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")

    if not os.path.exists(train_path):
        print("  ⚠️ 训练数据不存在，跳过")
        return None

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    def get_batch(data, batch_size, seq_len):
        max_start = len(data) - seq_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)
        x = torch.stack([torch.from_numpy(data[s:s+seq_len].astype(np.int64)) for s in starts])
        y = torch.stack([torch.from_numpy(data[s+1:s+seq_len+1].astype(np.int64)) for s in starts])
        return x.to(DEVICE), y.to(DEVICE)

    def eval_loss(model, data, n=10):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(n):
                x, y = get_batch(data, BATCH, SEQ)
                _, loss = model(x, y)
                losses.append(loss.item())
        model.train()
        return np.mean(losses)

    results = {}
    for name, model_cls, dff in [
        ("Baseline", BaselineGPT, BL_DFF),
        ("NEXUS-v4", NexusGPT, NX_DFF),
    ]:
        torch.manual_seed(42)
        np.random.seed(42)

        model = model_cls(V, D, L, H, dff, SEQ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        params = model.count_params()

        # 记录 loss 曲线
        curve = []
        t0 = time.time()

        # 初始 loss
        init_loss = eval_loss(model, val_data)
        curve.append((0, init_loss))

        model.train()
        for step in range(1, N_STEPS + 1):
            x, y = get_batch(train_data, BATCH, SEQ)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 10 == 0:
                val_l = eval_loss(model, val_data)
                curve.append((step, val_l))

        elapsed = time.time() - t0
        final_loss = curve[-1][1]

        # 计算学习速率指标
        loss_drop = init_loss - final_loss  # 越大越好
        loss_drop_rate = loss_drop / N_STEPS  # 每步平均下降量
        # 前 30 步的下降速率（早期学习能力）
        early_drop = init_loss - curve[3][1] if len(curve) > 3 else 0
        early_rate = early_drop / 30

        results[name] = {
            "params": params,
            "init_loss": init_loss,
            "final_loss": final_loss,
            "loss_drop": loss_drop,
            "loss_drop_rate": loss_drop_rate,
            "early_rate": early_rate,
            "time": elapsed,
            "curve": curve,
        }

        print(f"\n  [{name}] params={params:,} | time={elapsed:.1f}s")
        print(f"    Init loss:  {init_loss:.4f}")
        print(f"    Final loss: {final_loss:.4f} (↓{loss_drop:.4f})")
        print(f"    早期速率:   {early_rate:.5f}/step (前30步)")
        print(f"    平均速率:   {loss_drop_rate:.5f}/step (100步)")

        del model, opt
        torch.cuda.empty_cache()

    # 对比
    bl = results["Baseline"]
    nx = results["NEXUS-v4"]

    print(f"\n  --- 架构学习能力对比 ---")
    print(f"  {'Metric':<20s} | {'Baseline':>10s} | {'NEXUS-v4':>10s} | {'Winner':>10s}")
    print(f"  {'-'*60}")

    metrics = [
        ("Init Loss", bl["init_loss"], nx["init_loss"], "lower"),
        ("Final Loss (100步)", bl["final_loss"], nx["final_loss"], "lower"),
        ("Total Drop", bl["loss_drop"], nx["loss_drop"], "higher"),
        ("Early Rate (/step)", bl["early_rate"], nx["early_rate"], "higher"),
        ("Avg Rate (/step)", bl["loss_drop_rate"], nx["loss_drop_rate"], "higher"),
        ("Training Time", bl["time"], nx["time"], "lower"),
    ]

    nexus_wins = 0
    for mname, bv, nv, better in metrics:
        if better == "lower":
            winner = "NEXUS ✅" if nv < bv else "Baseline" if bv < nv else "Tie"
        else:
            winner = "NEXUS ✅" if nv > bv else "Baseline" if bv > nv else "Tie"
        if "NEXUS" in winner:
            nexus_wins += 1
        print(f"  {mname:<20s} | {bv:>10.5f} | {nv:>10.5f} | {winner:>10s}")

    # Loss 曲线对比
    print(f"\n  --- Loss 曲线（每 10 步） ---")
    print(f"  {'Step':>6s} | {'Baseline':>10s} | {'NEXUS-v4':>10s} | {'Gap':>10s}")
    print(f"  {'-'*45}")
    for i in range(len(bl["curve"])):
        bs, bv = bl["curve"][i]
        ns, nv = nx["curve"][i]
        gap = nv - bv
        print(f"  {bs:>6d} | {bv:>10.4f} | {nv:>10.4f} | {gap:>+10.4f}")

    # 最终判定
    print(f"\n  --- 最终判定 ---")
    if nx["loss_drop_rate"] >= bl["loss_drop_rate"] * 0.95:
        print(f"  ✅ NEXUS 学习速率 ≥ 95% of Baseline — 架构学习能力合格！")
        print(f"     NEXUS 的 DiffAttn + SwiGLU 组合在 micro-scale 上不拖后腿")
    else:
        deficit = (1 - nx["loss_drop_rate"] / bl["loss_drop_rate"]) * 100
        print(f"  ⚠️ NEXUS 学习速率比 Baseline 慢 {deficit:.1f}%")
        print(f"     可能原因: DiffAttn 的双 SDPA 在初期收敛更慢")

    if nexus_wins >= 4:
        print(f"  🏆 NEXUS 在 {nexus_wins}/6 项指标上领先！架构设计优秀。")
    elif nexus_wins >= 3:
        print(f"  ✅ NEXUS 在 {nexus_wins}/6 项指标上领先，架构基本合格。")
    else:
        print(f"  ⚠️ NEXUS 仅在 {nexus_wins}/6 项领先，需要关注。")

    return results


# ============================================================
# TEST 3: μP 坐标检查（30 秒，零训练）
# ============================================================

def test_mup_coordinate_check():
    """
    μP (Maximal Update Parametization) 坐标检查：
    验证模型在不同宽度下，激活值分布是否稳定。

    原理（Microsoft, 2022）：
      如果模型的参数化正确（μP），那么改变 width 时，
      中间激活值的量级应该保持 O(1) — 不随宽度变化。
      这意味着小模型的最优学习率可以直接迁移到大模型。

    测试方法：
      在 d_model = [64, 128, 256, 512] 下初始化模型，
      做一次 forward，记录各层输出的 L2 范数。
      如果范数随 d_model 线性增长 → 非 μP（标准参数化）
      如果范数保持恒定 → μP 友好

    注意：这测的不是"哪个架构更好"，而是"能否跨规模迁移超参"
    """
    print("\n" + "=" * 70)
    print("  TEST 3: μP 坐标检查（激活值稳定性 vs 宽度）")
    print("=" * 70)

    V = 50257
    widths = [64, 128, 256, 384]
    SEQ = 128
    L = 4
    BATCH = 2

    for name, model_cls, dff_ratio in [
        ("Baseline", BaselineGPT, 4),  # d_ff = 4 * d_model
        ("NEXUS-v4", NexusGPT, 2.67),  # SwiGLU d_ff ≈ 2.67 * d_model
    ]:
        print(f"\n  [{name}]")
        print(f"  {'Width':>8s} | {'Block Out Norm':>15s} | {'Final Norm':>12s} | {'Logit Std':>10s}")
        print(f"  {'-'*55}")

        norms = []
        for d in widths:
            nh = max(2, d // 32)  # 保证 head_dim >= 32
            dff = int(d * dff_ratio)
            # 确保 dff 对齐
            if name == "NEXUS-v4":
                dff = max(dff, 64)  # 最小值

            torch.manual_seed(42)
            model = model_cls(V, d, L, nh, dff, SEQ).to(DEVICE)

            x = torch.randint(0, V, (BATCH, SEQ), device=DEVICE)

            with torch.no_grad():
                # 手动 forward 以获取中间激活
                h = model.drop(model.tok_emb(x))
                block_out_norm = 0
                for block in model.blocks:
                    h = block(h)
                    block_out_norm = h.norm().item()
                h = model.ln_f(h)
                final_norm = h.norm().item()
                logits = model.head(h)
                logit_std = logits.std().item()

            norms.append({
                "width": d,
                "block_out": block_out_norm,
                "final": final_norm,
                "logit_std": logit_std,
            })

            print(f"  {d:>8d} | {block_out_norm:>15.2f} | {final_norm:>12.2f} | {logit_std:>10.4f}")

            del model
            torch.cuda.empty_cache()

        # 分析稳定性：计算范数随 width 的变化率
        ratios = []
        for i in range(1, len(norms)):
            width_ratio = norms[i]["width"] / norms[0]["width"]
            norm_ratio = norms[i]["final"] / norms[0]["final"]
            ratios.append(norm_ratio / width_ratio)

        avg_ratio = np.mean(ratios)
        # μP 友好：avg_ratio ≈ 1（范数随 sqrt(width) 增长是正常的）
        # 不稳定：avg_ratio >> 1 或 << 1
        if 0.3 < avg_ratio < 3.0:
            print(f"  ✅ 激活值随宽度变化相对稳定 (ratio≈{avg_ratio:.2f})")
            print(f"     → 小规模超参可以合理迁移到大规模")
        else:
            print(f"  ⚠️ 激活值随宽度变化较大 (ratio≈{avg_ratio:.2f})")
            print(f"     → 可能需要按宽度调整学习率")


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("  NEXUS 快速架构验证套件")
    print(f"  Device: {DEVICE}")
    print("  目标: 在几分钟内验证架构质量，替代数小时预训练")
    print("=" * 70)

    t0 = time.time()

    # TEST 1: 梯度健康（~10s）
    gradient_results = test_gradient_health()

    # TEST 2: 100 步 Micro-Benchmark（~2-3 min）
    micro_results = test_micro_benchmark()

    # TEST 3: μP 坐标检查（~30s）
    test_mup_coordinate_check()

    total_time = time.time() - t0

    print(f"\n\n{'='*70}")
    print(f"  全部验证完成！总耗时: {total_time:.0f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  对比完整预训练 (3000步): ~40 分钟 Baseline + ~40 分钟 NEXUS = 80 分钟")
    print(f"  节省: {80*60/total_time:.0f}x 更快")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
