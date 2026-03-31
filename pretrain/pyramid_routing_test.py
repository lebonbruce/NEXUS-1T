"""
W 快照 Content-Aware Routing — 金字塔假设验证 v3

关键改进：
  v2 用 output_norm 做 routing key → 信号太弱（权重≈50/50）
  v3 用 content fingerprint：每个 W 快照维护一个"内容指纹"
      （记录它在学习阶段见过的 token 嵌入均值）。
      
  Routing: query 和 fingerprint 的余弦相似度 → softmax → 选择 W

  这模拟了金字塔的实际运行方式：
  checkpoint 时同时保存 W 和一个"这段文本在讲什么"的摘要向量
"""

import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain"))
from models import TTTLinear

DEVICE = "cuda"


def main():
    print("=" * 70)
    print("  W 快照 Content-Aware Routing — 金字塔验证 v3")
    print("=" * 70)

    D = 256
    V = 256
    MBS = 16
    LR = 0.5

    # 正交嵌入
    torch.manual_seed(42)
    raw = torch.randn(V, D, device=DEVICE)
    embed_table, _ = torch.linalg.qr(raw.t())
    embed_table = embed_table.t()

    # 两组截然不同的模式 token
    A_ids = list(range(10, 20))    # A 模式 = token 10-19
    B_ids = list(range(100, 110))  # B 模式 = token 100-109
    noise_ids = list(range(150, 250))

    torch.manual_seed(42)
    np.random.seed(42)
    ttt = TTTLinear(D, mini_batch_size=MBS, ttt_lr=LR).to(DEVICE).eval()

    t0 = time.time()

    # 流式处理 + 记录每阶段的 fingerprint
    cum_grad = torch.zeros(D, D, device=DEVICE)
    snapshots = {}

    def process_phase(name, pattern_ids, n_tokens):
        nonlocal cum_grad
        fingerprint = torch.zeros(D, device=DEVICE)
        n_fp = 0

        for start in range(0, n_tokens, 1024):
            end = min(start + 1024, n_tokens)
            n = end - start
            usable = (n // MBS) * MBS
            if usable == 0:
                continue

            # 生成 token
            tokens = []
            for _ in range(usable):
                if np.random.random() < 0.5:
                    tokens.append(pattern_ids[np.random.randint(0, len(pattern_ids))])
                else:
                    tokens.append(noise_ids[np.random.randint(0, len(noise_ids))])

            x = embed_table[torch.tensor(tokens, device=DEVICE)].unsqueeze(0)
            fingerprint += x.squeeze(0).mean(dim=0)
            n_fp += 1

            with torch.no_grad():
                x_mb = x.view(1, usable // MBS, MBS, D)
                target = ttt.theta_proj(x_mb)
                W_current = ttt.W.weight - cum_grad
                pred = torch.matmul(x_mb, W_current.t())
                error = pred - target
                grad = torch.matmul(error.transpose(-1, -2), x_mb) / MBS
                grad = torch.clamp(grad, -1.0, 1.0)
                x_mean = x_mb.mean(dim=2)
                lr_mod = ttt.lr_gate(x_mean)
                eff_lr = ttt.ttt_base_lr * lr_mod.mean(-1, keepdim=True).unsqueeze(-1)
                cum_grad += (eff_lr * grad).float().sum(dim=1).squeeze(0)

        fingerprint = fingerprint / n_fp  # 平均嵌入 = 内容指纹
        snapshots[name] = {
            "grad": cum_grad.clone(),
            "fingerprint": fingerprint,
        }
        print(f"  {name}: W_norm={cum_grad.norm():.4f}, fp_norm={fingerprint.norm():.4f}")

    print("\n  [Phase 1] 学习模式 A (4K tokens)...")
    process_phase("phase1_A", A_ids, 4096)

    print("  [Phase 2] 学习模式 B (16K tokens)，冲刷 A...")
    process_phase("phase2_B", B_ids, 16384)

    # W 快照差异
    cos_w = F.cosine_similarity(
        snapshots["phase1_A"]["grad"].flatten().unsqueeze(0),
        snapshots["phase2_B"]["grad"].flatten().unsqueeze(0)
    ).item()
    cos_fp = F.cosine_similarity(
        snapshots["phase1_A"]["fingerprint"].unsqueeze(0),
        snapshots["phase2_B"]["fingerprint"].unsqueeze(0)
    ).item()
    print(f"\n  W 差异: cos={cos_w:.4f}")
    print(f"  Fingerprint 差异: cos={cos_fp:.4f}")

    # === Content-Aware Routing ===
    print(f"\n  === Content-Aware Routing ===")

    query_A = embed_table[A_ids].mean(dim=0)  # 模式 A 的"摘要"
    query_B = embed_table[B_ids].mean(dim=0)  # 模式 B 的"摘要"

    W_0 = ttt.W.weight

    # 计算 query 和每个 fingerprint 的相似度
    fps = torch.stack([s["fingerprint"] for s in snapshots.values()])  # [2, D]
    sim_A = F.cosine_similarity(query_A.unsqueeze(0), fps, dim=-1)  # [2]
    sim_B = F.cosine_similarity(query_B.unsqueeze(0), fps, dim=-1)

    temp = 0.05  # 低温 → 选择更锐利
    weight_A = F.softmax(sim_A / temp, dim=0)
    weight_B = F.softmax(sim_B / temp, dim=0)

    print(f"\n  Routing 权重（低温 softmax, τ={temp}）：")
    names = list(snapshots.keys())
    print(f"  查询模式 A:")
    for i, n in enumerate(names):
        marker = " ⭐" if weight_A[i] == weight_A.max() else ""
        print(f"    → {n}: sim={sim_A[i]:.4f}, weight={weight_A[i]:.4f}{marker}")

    print(f"  查询模式 B:")
    for i, n in enumerate(names):
        marker = " ⭐" if weight_B[i] == weight_B.max() else ""
        print(f"    → {n}: sim={sim_B[i]:.4f}, weight={weight_B[i]:.4f}{marker}")

    # 用 routing 后的加权 W 计算输出
    grads = torch.stack([s["grad"] for s in snapshots.values()])  # [2, D, D]

    # 模式 A 查询的聚合输出
    W_routed_A = W_0 - (weight_A.unsqueeze(-1).unsqueeze(-1) * grads).sum(dim=0)
    out_A_routed = torch.matmul(embed_table[A_ids], W_routed_A.t())

    # 模式 B 查询的聚合输出
    W_routed_B = W_0 - (weight_B.unsqueeze(-1).unsqueeze(-1) * grads).sum(dim=0)
    out_B_routed = torch.matmul(embed_table[B_ids], W_routed_B.t())

    # 对比：只用最新 W
    W_latest = W_0 - snapshots["phase2_B"]["grad"]
    out_A_latest = torch.matmul(embed_table[A_ids], W_latest.t())
    out_B_latest = torch.matmul(embed_table[B_ids], W_latest.t())

    # 只用 phase1 W
    W_p1 = W_0 - snapshots["phase1_A"]["grad"]
    out_A_p1 = torch.matmul(embed_table[A_ids], W_p1.t())
    out_B_p1 = torch.matmul(embed_table[B_ids], W_p1.t())

    print(f"\n  === 输出响应对比 ===")
    print(f"  {'策略':<30s} | {'A 响应':>8s} | {'B 响应':>8s}")
    print(f"  {'-'*50}")
    print(f"  {'只用 Phase1 W':<30s} | {out_A_p1.norm(dim=-1).mean():.4f} | {out_B_p1.norm(dim=-1).mean():.4f}")
    print(f"  {'只用最新 W (Phase2)':<30s} | {out_A_latest.norm(dim=-1).mean():.4f} | {out_B_latest.norm(dim=-1).mean():.4f}")
    print(f"  {'Content-Aware Routing':<30s} | {out_A_routed.norm(dim=-1).mean():.4f} | {out_B_routed.norm(dim=-1).mean():.4f}")

    # === 最终验证 ===
    print(f"\n{'='*70}")
    print(f"  最终验证")
    print(f"{'='*70}")

    a_prefers_p1 = weight_A[0] > weight_A[1]
    b_prefers_p2 = weight_B[1] > weight_B[0]

    print(f"\n  Routing 行为分析:")
    print(f"    查模式 A → {'Phase1 (正确！) ✅' if a_prefers_p1 else 'Phase2 ⚠️'}")
    print(f"      权重: P1={weight_A[0]:.4f}, P2={weight_A[1]:.4f}")
    print(f"    查模式 B → {'Phase2 (正确！) ✅' if b_prefers_p2 else 'Phase1 ⚠️'}")
    print(f"      权重: P1={weight_B[0]:.4f}, P2={weight_B[1]:.4f}")

    if a_prefers_p1 and b_prefers_p2:
        print(f"""
  🏆 金字塔假设完全验证通过！

  Content-Aware Routing 自动实现了：
    ✅ 查旧知识(A) → 选择存有 A 的快照
    ✅ 查新知识(B) → 选择存有 B 的快照

  这证明了分层压缩金字塔的核心机制可行：
    1. 不同阶段的 W 存储了不同信息（W cos={cos_w:.4f}）
    2. Fingerprint 可以区分不同阶段的内容（FP cos={cos_fp:.4f}）
    3. Content-aware routing 能按需检索正确的 W 快照

  方案 3 的工程实现路线图：
    Level 0: 滑动窗口 DiffAttn（局部 4K）
    Level 1: 每 1K token 保存 W 快照 + fingerprint
    Level 2: 每 100K token 合并 Level 1 快照
    Level 3: 全局 W（只保留最宏观的模式）
    Routing: query → fingerprint cosine → softmax → 加权 W
        """)
    else:
        w_diff = abs(weight_A[0] - weight_A[1])
        print(f"\n  routing 差异: {w_diff:.4f}")
        print(f"  需要更强的模式差异或更多训练步数。")

    print(f"\n  总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
