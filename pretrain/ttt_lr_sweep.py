"""
TTT 学习率 Sweep — 找到最优 ttt_lr

用 7 秒测试框架快速扫描不同 ttt_lr 下的在线学习斜率。
目标：找到使 loss 下降斜率最大（最负）的 ttt_lr。
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain"))
from models import NexusGPT, BaselineGPT

DEVICE = "cuda"


def measure_slope(model, data, V, seq_len, batch_size=2, n_trials=10, n_bins=16):
    """测量序列内 loss 的线性斜率。"""
    bin_size = seq_len // n_bins
    bin_losses = np.zeros(n_bins)

    model.eval()
    with torch.no_grad():
        for _ in range(n_trials):
            starts = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
            x = torch.stack([
                torch.from_numpy(data[s:s + seq_len].astype(np.int64))
                for s in starts
            ]).to(DEVICE)
            y = torch.stack([
                torch.from_numpy(data[s + 1:s + seq_len + 1].astype(np.int64))
                for s in starts
            ]).to(DEVICE)

            logits, _ = model(x)
            for b in range(n_bins):
                s, e = b * bin_size, (b + 1) * bin_size
                loss = F.cross_entropy(
                    logits[:, s:e, :].reshape(-1, V),
                    y[:, s:e].reshape(-1)
                ).item()
                bin_losses[b] += loss / n_trials

    slope, _ = np.polyfit(np.arange(n_bins), bin_losses, 1)
    return slope, bin_losses


def main():
    print("=" * 70)
    print("  TTT 学习率 Sweep — 找最优在线适应速率")
    print("=" * 70)

    # 数据
    data_dir = None
    for base in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain", "data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        os.path.join("pretrain", "data"),
    ]:
        if os.path.exists(os.path.join(base, "val.bin")):
            data_dir = base
            break

    if data_dir is None:
        print("  ❌ 数据未找到")
        return

    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    V = 50257
    D = 512
    L = 4
    H = 8
    SEQ = 1024
    NX_DFF = int(D * 4 * 2 / 3)
    BL_DFF = D * 4

    # 先测 Baseline 的斜率作为基准
    torch.manual_seed(42)
    np.random.seed(42)
    baseline = BaselineGPT(V, D, L, H, BL_DFF, SEQ).to(DEVICE)
    bl_slope, _ = measure_slope(baseline, val_data, V, SEQ)
    del baseline
    torch.cuda.empty_cache()
    print(f"\n  Baseline 斜率 (参考): {bl_slope:+.6f}")

    # Sweep ttt_lr
    lr_values = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

    print(f"\n  {'ttt_lr':>10s} | {'Slope':>12s} | {'vs BL':>10s} | {'Status'}")
    print(f"  {'-'*55}")

    best_lr = None
    best_slope = float("inf")

    for lr in lr_values:
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.empty_cache()

        # 需要修改 TTTLinear 的 lr——通过构建时传参
        # 但当前 NexusBlock 硬编码了 ttt_lr=1e-3
        # 所以我们需要在构建后手动修改
        model = NexusGPT(V, D, L, H, NX_DFF, SEQ).to(DEVICE)

        # 手动修改所有 TTT 层的学习率
        for block in model.blocks:
            if hasattr(block, 'ttt') and block.use_ttt:
                block.ttt.ttt_base_lr = lr

        slope, bins = measure_slope(model, val_data, V, SEQ)

        diff = slope - bl_slope
        nan_detected = np.isnan(slope) or any(np.isnan(bins))

        if nan_detected:
            status = "💥 NaN! (lr 太大)"
        elif slope < best_slope and not nan_detected:
            best_slope = slope
            best_lr = lr
            status = "⭐ 当前最优"
        elif slope < bl_slope:
            status = "✅ 优于 Baseline"
        else:
            status = "⚠️ 不如 Baseline"

        print(f"  {lr:>10.0e} | {slope:>+12.6f} | {diff:>+10.6f} | {status}")

        del model
        torch.cuda.empty_cache()

        if nan_detected:
            print(f"  ↑ lr={lr} 导致 NaN，更大的 lr 不用试了")
            break

    print(f"\n  {'='*55}")
    print(f"  最优 ttt_lr: {best_lr:.0e} (slope={best_slope:+.6f})")
    print(f"  Baseline:    {bl_slope:+.6f}")
    print(f"  TTT 优势:    {best_slope - bl_slope:+.6f}")

    if best_slope < bl_slope:
        improvement = (bl_slope - best_slope) / abs(bl_slope) * 100
        print(f"  ✅ 最优 TTT 的在线适应速度比 Baseline 好 {improvement:.0f}%!")
    else:
        print(f"  ⚠️ 所有 ttt_lr 配置下 TTT 都不优于 Baseline")

    print(f"\n  建议: 在 models.py TTTLinear.__init__ 中将 ttt_lr 默认值设为 {best_lr:.0e}")


if __name__ == "__main__":
    main()
