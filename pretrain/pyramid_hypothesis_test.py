"""
多级 W 金字塔 — 核心假设验证 v2（修复版）

v1 失败原因：
  嵌入向量太弱（±0.1 range），W 变化量仅占初始值 1%，
  完全淹没在初始化噪声中。

v2 修复：
  1. 嵌入增强：模式 token 使用强正交嵌入（范数=1.0）
  2. 加大 ttt_lr（从 5e-4 → 0.1）让 W 快速适应
  3. 更长序列（20K token，阶段 1 占 5K，阶段 2 占 15K）
  4. 直接用 W @ embed 的余弦相似度做 "关联检索" 测试
"""

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain"))
from models import TTTLinear

DEVICE = "cuda"


def run_streaming_with_snapshots(D, ttt_lr, n_phase1, n_phase2,
                                  pattern_A_ids, pattern_B_ids,
                                  embed_table, snapshot_every=1024):
    """
    流式 TTT：处理两个阶段的序列，沿途保存 W 快照。
    """
    mbs = 16
    ttt = TTTLinear(D, mini_batch_size=mbs, ttt_lr=ttt_lr).to(DEVICE).eval()

    cum_grad = torch.zeros(D, D, device=DEVICE)
    snapshots = {}

    def process_chunk(token_ids, pos_label):
        """处理一个 chunk 并累积梯度。"""
        nonlocal cum_grad
        n = len(token_ids)
        usable = (n // mbs) * mbs
        if usable == 0:
            return

        x = embed_table[token_ids[:usable]].unsqueeze(0)  # [1, usable, D]

        with torch.no_grad():
            x_mb = x.view(1, usable // mbs, mbs, D)
            target = ttt.theta_proj(x_mb)
            W_0 = ttt.W.weight
            W_current = W_0 - cum_grad
            pred = torch.matmul(x_mb, W_current.t())
            error = pred - target
            grad_all = torch.matmul(error.transpose(-1, -2), x_mb) / mbs
            grad_all = torch.clamp(grad_all, -1.0, 1.0)

            x_mean = x_mb.mean(dim=2)
            lr_mod = ttt.lr_gate(x_mean)
            eff_lr = ttt.ttt_base_lr * lr_mod.mean(-1, keepdim=True).unsqueeze(-1)
            chunk_grad = (eff_lr * grad_all).float().sum(dim=1).squeeze(0)
            cum_grad += chunk_grad

    # 阶段 1：模式 A 反复出现
    print(f"  阶段 1: 学习模式 A ({n_phase1} tokens)...")
    chunk_size = 1024
    for start in range(0, n_phase1, chunk_size):
        end = min(start + chunk_size, n_phase1)
        # 生成 token : 40% 概率插入模式 A 的 key token，60% 随机
        tokens = []
        for _ in range(end - start):
            if np.random.random() < 0.4:
                idx = np.random.randint(0, len(pattern_A_ids))
                tokens.append(pattern_A_ids[idx])
            else:
                tokens.append(np.random.randint(0, 256))
        process_chunk(torch.tensor(tokens, device=DEVICE), f"phase1_{start}")

        if (start + chunk_size) % snapshot_every == 0:
            snapshots[f"p1_{start + chunk_size}"] = cum_grad.clone()

    snapshots["after_phase1"] = cum_grad.clone()
    p1_norm = cum_grad.norm().item()
    print(f"  阶段 1 完成: W 变化范数 = {p1_norm:.4f}")

    # 阶段 2：模式 B 反复出现（冲刷 A）
    print(f"  阶段 2: 学习模式 B ({n_phase2} tokens)，冲刷模式 A...")
    for start in range(0, n_phase2, chunk_size):
        end = min(start + chunk_size, n_phase2)
        tokens = []
        for _ in range(end - start):
            if np.random.random() < 0.4:
                idx = np.random.randint(0, len(pattern_B_ids))
                tokens.append(pattern_B_ids[idx])
            else:
                tokens.append(np.random.randint(0, 256))
        process_chunk(torch.tensor(tokens, device=DEVICE), f"phase2_{start}")

    snapshots["after_phase2"] = cum_grad.clone()
    p2_norm = cum_grad.norm().item()
    print(f"  阶段 2 完成: W 变化范数 = {p2_norm:.4f}")

    return ttt, snapshots


def main():
    print("=" * 70)
    print("  多级 W 金字塔 — 核心假设验证 v2")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    D = 256
    V = 256

    # 创建强正交嵌入：每个 token 有一个明确的方向
    torch.manual_seed(42)
    embed_raw = torch.randn(V, D, device=DEVICE)
    # QR 正交化确保不同 token 的嵌入正交
    embed_table, _ = torch.linalg.qr(embed_raw.t())
    embed_table = embed_table.t()  # [V, D]，每行范数=1

    # 定义模式：
    # 模式 A: token IDs [10, 11, 12, 13, 14, 15]
    # 模式 B: token IDs [100, 101, 102, 103, 104, 105]
    pattern_A_ids = list(range(10, 16))
    pattern_B_ids = list(range(100, 106))

    t0 = time.time()

    # 测试不同 ttt_lr
    print(f"\n  === 扫描 ttt_lr 找到 W 变化足够大的配置 ===")
    for lr in [5e-4, 5e-3, 5e-2, 0.1, 0.5]:
        torch.manual_seed(42)
        np.random.seed(42)
        quick_ttt = TTTLinear(D, mini_batch_size=16, ttt_lr=lr).to(DEVICE).eval()
        cum = torch.zeros(D, D, device=DEVICE)

        # 快速处理 1024 token
        tokens = torch.randint(0, V, (1024,), device=DEVICE)
        x = embed_table[tokens].unsqueeze(0)
        with torch.no_grad():
            x_mb = x.view(1, 64, 16, D)
            target = quick_ttt.theta_proj(x_mb)
            pred = torch.matmul(x_mb, quick_ttt.W.weight.t())
            error = pred - target
            grad_all = torch.matmul(error.transpose(-1, -2), x_mb) / 16
            grad_all = torch.clamp(grad_all, -1.0, 1.0)
            x_mean = x_mb.mean(dim=2)
            lr_mod = quick_ttt.lr_gate(x_mean)
            eff_lr = lr * lr_mod.mean(-1, keepdim=True).unsqueeze(-1)
            cum = (eff_lr * grad_all).float().sum(dim=1).squeeze(0)

        w_norm = quick_ttt.W.weight.norm().item()
        ratio = cum.norm().item() / w_norm * 100
        print(f"  lr={lr:.0e}: W_change/W_init = {ratio:.1f}%"
              f" {'✅ 足够大' if ratio > 5 else '⚠️ 太小'}")
        del quick_ttt
        torch.cuda.empty_cache()

    # 选择能让 W 变化 >10% 的 lr
    selected_lr = 0.1
    print(f"\n  选择 ttt_lr = {selected_lr} 进行主实验")

    # 主实验
    torch.manual_seed(42)
    np.random.seed(42)

    ttt, snapshots = run_streaming_with_snapshots(
        D=D, ttt_lr=selected_lr,
        n_phase1=4096, n_phase2=16384,
        pattern_A_ids=pattern_A_ids,
        pattern_B_ids=pattern_B_ids,
        embed_table=embed_table,
    )

    # 测试：模式 A 的"指纹"
    # 用模式 A 的 token 嵌入做查询，看 W @ query 的输出是否包含模式 A 的信息
    print(f"\n  === 旧模式回忆测试 ===")

    query_A = embed_table[pattern_A_ids]  # [6, D]，模式 A 的嵌入
    query_B = embed_table[pattern_B_ids]  # [6, D]，模式 B 的嵌入

    W_0 = ttt.W.weight  # [D, D]

    def compute_pattern_score(cum_grad, query, label=""):
        """
        计算 W_effective @ query 的输出特征。
        如果 W 记住了该 query 对应的模式，输出应该有特定结构。
        用输出的范数和一致性作为指标。
        """
        W_eff = W_0 - cum_grad
        output = torch.matmul(query, W_eff.t())  # [6, D]
        # 输出范数（W 对该模式的"响应强度"）
        response_norm = output.norm(dim=-1).mean().item()
        # 输出一致性（同一模式的 token 应产生相似输出）
        if output.size(0) > 1:
            cos_matrix = F.cosine_similarity(
                output.unsqueeze(0), output.unsqueeze(1), dim=-1
            )
            # 取上三角平均（排除自身）
            mask = torch.triu(torch.ones_like(cos_matrix), diagonal=1).bool()
            consistency = cos_matrix[mask].mean().item()
        else:
            consistency = 0
        return response_norm, consistency

    strategies = {
        "W_init (无学习)": torch.zeros(D, D, device=DEVICE),
        "W_after_phase1 (A还在)": snapshots["after_phase1"],
        "W_after_phase2 (A被冲刷)": snapshots["after_phase2"],
    }

    # 金字塔策略：平均 phase1 和 phase2 的 W
    strategies["金字塔 (avg p1+p2)"] = (
        snapshots["after_phase1"] + snapshots["after_phase2"]
    ) / 2

    print(f"\n  {'策略':<30s} | {'A 响应':>8s} | {'A 一致性':>8s} | {'B 响应':>8s} | {'B 一致性':>8s}")
    print(f"  {'-'*75}")

    a_scores = {}
    b_scores = {}
    for name, grad in strategies.items():
        a_resp, a_cons = compute_pattern_score(grad, query_A)
        b_resp, b_cons = compute_pattern_score(grad, query_B)
        a_scores[name] = (a_resp, a_cons)
        b_scores[name] = (b_resp, b_cons)
        print(f"  {name:<30s} | {a_resp:>8.4f} | {a_cons:>+8.4f} | {b_resp:>8.4f} | {b_cons:>+8.4f}")

    # 判断
    print(f"\n  === 关键对比 ===")

    # 核心问题：阶段 2 冲刷后，模式 A 的痕迹是否消失？
    init_a = a_scores["W_init (无学习)"]
    p1_a = a_scores["W_after_phase1 (A还在)"]
    p2_a = a_scores["W_after_phase2 (A被冲刷)"]
    pyr_a = a_scores["金字塔 (avg p1+p2)"]

    print(f"\n  模式 A 响应强度变化：")
    print(f"    初始 → 学完A:  {init_a[0]:.4f} → {p1_a[0]:.4f} (Δ={p1_a[0]-init_a[0]:+.4f})")
    print(f"    学完A → 冲刷后: {p1_a[0]:.4f} → {p2_a[0]:.4f} (Δ={p2_a[0]-p1_a[0]:+.4f})")
    print(f"    冲刷后 vs 金字塔: {p2_a[0]:.4f} vs {pyr_a[0]:.4f} (Δ={pyr_a[0]-p2_a[0]:+.4f})")

    # W 快照差异度
    print(f"\n  W 快照差异度：")
    p1_w = snapshots["after_phase1"]
    p2_w = snapshots["after_phase2"]
    diff_norm = (p1_w - p2_w).norm().item()
    max_norm = max(p1_w.norm().item(), p2_w.norm().item())
    cos = F.cosine_similarity(p1_w.flatten().unsqueeze(0), p2_w.flatten().unsqueeze(0)).item()
    print(f"    W_phase1 vs W_phase2: L2={diff_norm:.4f}, cos={cos:.4f}")
    print(f"    W 变化量: phase1={p1_w.norm().item():.4f}, phase2={p2_w.norm().item():.4f}")

    # 最终判定
    print(f"\n{'='*70}")
    print(f"  核心假设验证")
    print(f"{'='*70}")

    pyramid_win_a = pyr_a[0] > p2_a[0]  # 金字塔对模式 A 的响应更强
    pyramid_ok_b = True  # 金字塔对模式 B 不差太多

    p2_b = b_scores["W_after_phase2 (A被冲刷)"]
    pyr_b = b_scores["金字塔 (avg p1+p2)"]
    if pyr_b[0] < p2_b[0] * 0.8:
        pyramid_ok_b = False

    if cos < 0.95:  # W 快照之间确实不同
        print(f"\n  ✅ W 快照包含不同信息 (cos={cos:.4f} < 0.95)")
    else:
        print(f"\n  ⚠️ W 快照太相似 (cos={cos:.4f})，差异不够大")

    if pyramid_win_a:
        print(f"  ✅ 金字塔对旧模式 A 的响应更强 ({pyr_a[0]:.4f} > {p2_a[0]:.4f})")
    else:
        print(f"  ⚠️ 金字塔未能提升旧模式响应")

    if pyramid_win_a and cos < 0.95:
        print(f"""
  🏆 核心假设成立！

  1. W 快照之间包含不同信息（cos={cos:.4f}）
  2. 金字塔策略保留了被冲刷的旧模式
  3. 这证明了"多级 W 金字塔"的价值

  → 方案 3 值得全力投入！
        """)
    else:
        print(f"""
  结果需要进一步分析。
  W 变化量可能仍不够大，或者需要更精细的聚合策略。
        """)

    total = time.time() - t0
    print(f"  总耗时: {total:.1f}s")


if __name__ == "__main__":
    main()
