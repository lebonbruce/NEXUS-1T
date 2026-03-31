"""
NEXUS Scale-Up 验证 — 全组件打开，多规模对比

目标：用梯度检查 + 100 步 micro-benchmark 验证：
  当 d_model 足够大、seq_len 足够长时，NEXUS 的全组件是否开始超越 Baseline？

测试规模：
  Tiny:   d=128, seq=256    → DiffAttn + SwiGLU (TTT OFF, MLA OFF)
  Small:  d=256, seq=512    → DiffAttn + SwiGLU (TTT OFF, MLA OFF)
  Medium: d=512, seq=1024   → DiffAttn + MLA 2x + TTT ON + SwiGLU  ← 所有组件启用！
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
from models import BaselineGPT, NexusGPT, get_mla_compression, should_enable_ttt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_data_dir():
    """找到训练数据目录。"""
    for base in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain", "data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        os.path.join("pretrain", "data"),
    ]:
        if os.path.exists(os.path.join(base, "train.bin")):
            return base
    return None


def get_batch(data, batch_size, seq_len):
    """从数据中随机采样 batch。"""
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = torch.stack([torch.from_numpy(data[s:s+seq_len].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(data[s+1:s+seq_len+1].astype(np.int64)) for s in starts])
    return x.to(DEVICE), y.to(DEVICE)


def eval_loss(model, data, batch_size, seq_len, n=10):
    """评估验证 loss。"""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n):
            x, y = get_batch(data, batch_size, seq_len)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return np.mean(losses)


def run_scale_test(d_model, n_layers, n_heads, seq_len, batch_size, n_steps,
                   train_data, val_data, label):
    """
    在指定规模下对比 Baseline 和 NEXUS 的 100 步学习能力。
    """
    V = 50257
    bl_dff = d_model * 4       # Baseline GELU: d_ff = 4 * d_model
    # SwiGLU 3 矩阵等效参数量: 3 * nx_dff * d = 2 * bl_dff * d → nx_dff = 2/3 * bl_dff
    nx_dff = int(d_model * 4 * 2 / 3)

    LR = 6e-4

    # 显示配置
    mla = get_mla_compression(d_model)
    ttt = should_enable_ttt(d_model, seq_len)
    print(f"\n  === {label} (d={d_model}, L={n_layers}, seq={seq_len}) ===")
    print(f"  组件状态: DiffAttn=ON | MLA={'OFF' if mla==1 else f'{mla}x'} | TTT={'ON ✅' if ttt else 'OFF'} | SwiGLU=ON")

    results = {}
    for name, model_cls, dff in [
        ("Baseline", BaselineGPT, bl_dff),
        ("NEXUS", NexusGPT, nx_dff),
    ]:
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.empty_cache()

        model = model_cls(V, d_model, n_layers, n_heads, dff, seq_len).to(DEVICE)
        params = model.count_params()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        # 初始 loss
        init_loss = eval_loss(model, val_data, batch_size, seq_len, n=5)

        # 训练 n_steps 步
        curve = [(0, init_loss)]
        t0 = time.time()
        model.train()

        for step in range(1, n_steps + 1):
            x, y = get_batch(train_data, batch_size, seq_len)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 10 == 0:
                val_l = eval_loss(model, val_data, batch_size, seq_len, n=5)
                curve.append((step, val_l))

        elapsed = time.time() - t0
        final_loss = curve[-1][1]
        drop = init_loss - final_loss
        rate = drop / n_steps

        results[name] = {
            "params": params,
            "init": init_loss,
            "final": final_loss,
            "drop": drop,
            "rate": rate,
            "time": elapsed,
            "curve": curve,
            "tok_per_s": batch_size * seq_len * n_steps / elapsed,
        }

        print(f"    [{name:>8s}] {params/1e6:5.1f}M | init={init_loss:.4f} → final={final_loss:.4f} | "
              f"drop={drop:.4f} rate={rate:.5f}/step | {elapsed:.0f}s | {results[name]['tok_per_s']:.0f} tok/s")

        del model, opt
        torch.cuda.empty_cache()

    # 对比
    bl = results["Baseline"]
    nx = results["NEXUS"]
    gap = nx["final"] - bl["final"]
    rate_ratio = nx["rate"] / bl["rate"] if bl["rate"] > 0 else 0

    if nx["rate"] > bl["rate"]:
        verdict = f"✅ NEXUS 学习更快！ rate={rate_ratio:.2f}x Baseline"
    elif rate_ratio > 0.95:
        verdict = f"≈ 追平 ({rate_ratio:.2f}x)"
    else:
        deficit = (1 - rate_ratio) * 100
        verdict = f"⚠️ NEXUS 慢 {deficit:.1f}%"

    print(f"    对比: gap={gap:+.4f} | {verdict}")

    return {
        "label": label,
        "d_model": d_model,
        "seq_len": seq_len,
        "baseline": bl,
        "nexus": nx,
        "gap": gap,
        "rate_ratio": rate_ratio,
    }


def main():
    print("=" * 70)
    print("  NEXUS Scale-Up 验证 — 全组件逐步开启对比")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    DATA_DIR = find_data_dir()
    if DATA_DIR is None:
        print("  ❌ 训练数据未找到！")
        return

    train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")

    # 多规模对比（从小到大，组件逐步开启）
    scales = [
        # (d_model, n_layers, n_heads, seq_len, batch, steps, label)
        (128, 4, 4, 256, 16, 100,
         "Tiny (DiffAttn+SwiGLU only)"),
        (256, 4, 8, 512, 8, 100,
         "Small (DiffAttn+SwiGLU only)"),
        (512, 4, 8, 1024, 4, 100,
         "Medium (ALL ON: DiffAttn+MLA+TTT+SwiGLU)"),
    ]

    all_results = []
    t0 = time.time()

    for d, nl, nh, seq, bs, steps, label in scales:
        result = run_scale_test(d, nl, nh, seq, bs, steps, train_data, val_data, label)
        all_results.append(result)

    total_time = time.time() - t0

    # 总结
    print(f"\n\n{'='*70}")
    print(f"  Scale-Up 趋势总结")
    print(f"{'='*70}")
    print(f"  {'Scale':<40s} | {'Rate Ratio':>12s} | {'Gap':>8s} | {'Components'}")
    print(f"  {'-'*90}")

    for r in all_results:
        mla = get_mla_compression(r["d_model"])
        ttt = should_enable_ttt(r["d_model"], r["seq_len"])
        comps = f"DiffAttn+SwiGLU"
        if mla > 1:
            comps += f"+MLA{mla}x"
        if ttt:
            comps += "+TTT"

        ratio = r["rate_ratio"]
        if ratio > 1.0:
            status = f"{ratio:.2f}x ✅✅"
        elif ratio > 0.95:
            status = f"{ratio:.2f}x ≈"
        else:
            status = f"{ratio:.2f}x ⚠️"

        print(f"  d={r['d_model']:>4d} seq={r['seq_len']:>5d} ({r['baseline']['params']/1e6:.1f}M)"
              f" | {status:>12s} | {r['gap']:>+8.4f} | {comps}")

    # 关键判断
    print(f"\n  --- 关键判断 ---")
    if len(all_results) >= 3:
        ratios = [r["rate_ratio"] for r in all_results]
        if ratios[-1] > ratios[0]:
            print(f"  ✅ 随规模增大，NEXUS 相对 Baseline 的学习速率在改善！")
            print(f"     Tiny: {ratios[0]:.2f}x → Medium: {ratios[-1]:.2f}x")
            print(f"     这说明 NEXUS 的组件在更大规模下确实更有优势")
            if ratios[-1] > 1.0:
                print(f"  🏆 在 Medium 规模下 NEXUS 已经超越 Baseline！")
        else:
            print(f"  ⚠️ 规模增大后 NEXUS 相对优势没有改善")
            print(f"     Tiny: {ratios[0]:.2f}x → Medium: {ratios[-1]:.2f}x")

    print(f"\n  总耗时: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
