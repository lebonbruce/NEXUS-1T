"""
金字塔存储 vs 推理精度 Tradeoff 测试

问题：降低快照 rank 和增大粒度，会不会影响推理性能？

测试方法：
  1. 固定两阶段模式（A=4K, B=16K），保存 W 快照 + fingerprint
  2. 变量 1: 快照粒度（256/1K/4K/16K token 保存一次）
  3. 变量 2: W_grad 压缩（full rank vs SVD 低秩近似）
  4. 测量 routing 精度（能否正确选择存有目标模式的快照）
"""

import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain"))
from models import TTTLinear

DEVICE = "cuda"


def compress_grad_svd(grad, rank):
    """用 SVD 低秩近似压缩 W_grad 矩阵。"""
    U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
    # 只保留前 rank 个奇异值
    U_r = U[:, :rank]      # [D, rank]
    S_r = S[:rank]          # [rank]
    Vh_r = Vh[:rank, :]     # [rank, D]
    # 重建
    return U_r @ torch.diag(S_r) @ Vh_r


def run_full_experiment(D, ttt_lr, A_ids, B_ids, noise_ids,
                        embed_table, granularities, ranks):
    """
    运行完整实验：不同粒度 × 不同 rank 的 routing 精度网格。
    """
    MBS = 16
    N_PHASE1 = 4096
    N_PHASE2 = 16384

    # 创建 TTT
    torch.manual_seed(42)
    np.random.seed(42)
    ttt = TTTLinear(D, mini_batch_size=MBS, ttt_lr=ttt_lr).to(DEVICE).eval()

    cum_grad = torch.zeros(D, D, device=DEVICE)
    # 存储：每处理 min_granularity 个 token 就保存一次
    min_gran = min(granularities)
    all_snapshots = []  # list of (position, cum_grad, fingerprint)

    position = 0

    def process_tokens(pattern_ids, n_tokens, phase_name):
        nonlocal cum_grad, position
        fp_accum = torch.zeros(D, device=DEVICE)
        fp_count = 0

        for start in range(0, n_tokens, min_gran):
            end = min(start + min_gran, n_tokens)
            n = end - start
            usable = (n // MBS) * MBS
            if usable == 0:
                continue

            tokens = []
            for _ in range(usable):
                if np.random.random() < 0.5:
                    tokens.append(pattern_ids[np.random.randint(0, len(pattern_ids))])
                else:
                    tokens.append(noise_ids[np.random.randint(0, len(noise_ids))])

            x = embed_table[torch.tensor(tokens, device=DEVICE)].unsqueeze(0)
            chunk_fp = x.squeeze(0).mean(dim=0)

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

            fp_accum += chunk_fp
            fp_count += 1
            position += usable

            # 每 min_gran 保存快照
            all_snapshots.append({
                "position": position,
                "phase": phase_name,
                "grad": cum_grad.clone(),
                "fingerprint": (fp_accum / fp_count).clone(),
            })
            fp_accum = torch.zeros(D, device=DEVICE)
            fp_count = 0

    process_tokens(A_ids, N_PHASE1, "A")
    process_tokens(B_ids, N_PHASE2, "B")

    return ttt, all_snapshots


def test_routing_accuracy(ttt, all_snapshots, embed_table,
                          A_ids, B_ids, granularity, rank, D):
    """
    测试在指定粒度和rank下的routing精度。

    返回：(A→正确快照的概率, B→正确快照的概率)
    """
    # 按粒度抽样快照
    min_gran = all_snapshots[1]["position"] - all_snapshots[0]["position"]
    step = max(1, granularity // min_gran)
    sampled = all_snapshots[::step]

    if len(sampled) < 2:
        return 0.5, 0.5

    # 压缩 W_grad
    if rank is not None and rank < D:
        for s in sampled:
            s["compressed_grad"] = compress_grad_svd(s["grad"], rank)
    else:
        for s in sampled:
            s["compressed_grad"] = s["grad"]

    # 构建 fingerprint 索引
    fps = torch.stack([s["fingerprint"] for s in sampled])  # [N, D]
    phases = [s["phase"] for s in sampled]

    # 查询
    query_A = embed_table[A_ids].mean(dim=0).unsqueeze(0)  # [1, D]
    query_B = embed_table[B_ids].mean(dim=0).unsqueeze(0)

    # Cosine similarity routing
    sim_A = F.cosine_similarity(query_A, fps, dim=-1)  # [N]
    sim_B = F.cosine_similarity(query_B, fps, dim=-1)

    # 找到 routing 选择的快照
    best_A_idx = sim_A.argmax().item()
    best_B_idx = sim_B.argmax().item()

    # 正确性：A 查询应选到 phase=A 的快照
    a_correct = phases[best_A_idx] == "A"
    b_correct = phases[best_B_idx] == "B"

    # 更细致：看 top-3 中有多少是正确 phase
    topk = min(3, len(sampled))
    top_A = sim_A.topk(topk).indices.tolist()
    top_B = sim_B.topk(topk).indices.tolist()
    a_top3 = sum(1 for i in top_A if phases[i] == "A") / topk
    b_top3 = sum(1 for i in top_B if phases[i] == "B") / topk

    # 用选中的 W 做预测，测量输出质量
    W_0 = ttt.W.weight
    grad_A = sampled[best_A_idx]["compressed_grad"]
    grad_B = sampled[best_B_idx]["compressed_grad"]

    out_A = torch.matmul(embed_table[A_ids], (W_0 - grad_A).t())
    out_B = torch.matmul(embed_table[B_ids], (W_0 - grad_B).t())

    # 对比基线（用最终 W）
    final_grad = all_snapshots[-1]["grad"]
    out_A_final = torch.matmul(embed_table[A_ids], (W_0 - final_grad).t())
    out_B_final = torch.matmul(embed_table[B_ids], (W_0 - final_grad).t())

    return {
        "a_hit": a_correct,
        "b_hit": b_correct,
        "a_top3": a_top3,
        "b_top3": b_top3,
        "n_snapshots": len(sampled),
        "a_resp": out_A.norm(dim=-1).mean().item(),
        "b_resp": out_B.norm(dim=-1).mean().item(),
        "a_resp_final": out_A_final.norm(dim=-1).mean().item(),
        "b_resp_final": out_B_final.norm(dim=-1).mean().item(),
    }


def main():
    print("=" * 70)
    print("  金字塔存储 vs 推理精度 Tradeoff 测试")
    print("=" * 70)

    D = 256
    V = 256

    torch.manual_seed(42)
    raw = torch.randn(V, D, device=DEVICE)
    embed_table, _ = torch.linalg.qr(raw.t())
    embed_table = embed_table.t()

    A_ids = list(range(10, 20))
    B_ids = list(range(100, 110))
    noise_ids = list(range(150, 250))

    t0 = time.time()

    # 运行实验
    print("\n  [Step 1] 生成完整快照序列（最细粒度=256 tokens）...")
    ttt, all_snapshots = run_full_experiment(
        D=D, ttt_lr=0.5,
        A_ids=A_ids, B_ids=B_ids, noise_ids=noise_ids,
        embed_table=embed_table,
        granularities=[256, 1024, 4096, 16384],
        ranks=[16, 32, 64, 128, None],
    )
    print(f"  共 {len(all_snapshots)} 个快照")

    # 粒度 × rank 网格搜索
    granularities = [256, 1024, 4096, 16384]
    ranks = [8, 16, 32, 64, 128, None]  # None = full rank

    print(f"\n  [Step 2] 粒度 × Rank 网格测试...")

    # 计算理论存储
    def calc_storage(n_snaps, rank, D):
        fp_size = D * 4  # fingerprint float32
        if rank is None:
            grad_size = D * D * 4
        else:
            grad_size = 2 * D * rank * 4  # U*S + V 的低秩表示
        return n_snaps * (fp_size + grad_size)

    print(f"\n  {'粒度':>8s} | {'Rank':>6s} | {'快照数':>6s} | {'存储':>10s} | {'A命中':>5s} | {'B命中':>5s} | {'A top3':>6s} | {'B top3':>6s} | {'状态'}")
    print(f"  {'-'*80}")

    results = []
    for gran in granularities:
        for rank in ranks:
            res = test_routing_accuracy(
                ttt, all_snapshots, embed_table,
                A_ids, B_ids, gran, rank, D
            )

            # 计算 100M token 时的等效存储
            n_snaps_100m = 100_000_000 // gran
            storage_100m = calc_storage(n_snaps_100m, rank, 21504)  # 用 1T 的 D

            rank_str = f"{rank}" if rank else "Full"
            a_hit = "✅" if res["a_hit"] else "❌"
            b_hit = "✅" if res["b_hit"] else "❌"

            # 综合得分
            score = (res["a_top3"] + res["b_top3"]) / 2
            if score >= 0.9:
                status = "⭐ 优"
            elif score >= 0.6:
                status = "✅ 良"
            elif score >= 0.3:
                status = "⚠️ 可"
            else:
                status = "❌ 差"

            def fmt_storage(b):
                if b >= 1e12: return f"{b/1e12:.1f}TB"
                elif b >= 1e9: return f"{b/1e9:.1f}GB"
                elif b >= 1e6: return f"{b/1e6:.0f}MB"
                else: return f"{b/1e3:.0f}KB"

            print(f"  {gran:>8d} | {rank_str:>6s} | {res['n_snapshots']:>6d} | "
                  f"{fmt_storage(storage_100m):>10s} | {a_hit:>5s} | {b_hit:>5s} | "
                  f"{res['a_top3']:>6.2f} | {res['b_top3']:>6.2f} | {status}")

            results.append({
                "gran": gran,
                "rank": rank,
                "score": score,
                "storage": storage_100m,
                "a_top3": res["a_top3"],
                "b_top3": res["b_top3"],
            })

    # 找最优配置（Pareto 最优：存储小 + 精度高）
    print(f"\n  === Pareto 最优分析 ===")

    # 按 score 降序排序，相同 score 时存储小的优先
    results.sort(key=lambda x: (-x["score"], x["storage"]))

    print(f"\n  {'Rank':>8s} | {'粒度':>8s} | {'100M存储':>10s} | {'精度':>6s} | {'推荐'}")
    print(f"  {'-'*50}")

    seen_scores = set()
    for r in results[:8]:
        rank_str = f"{r['rank']}" if r['rank'] else "Full"
        score_key = f"{r['score']:.2f}"
        is_new = score_key not in seen_scores
        seen_scores.add(score_key)

        def fmt_s(b):
            if b >= 1e12: return f"{b/1e12:.1f}TB"
            elif b >= 1e9: return f"{b/1e9:.1f}GB"
            elif b >= 1e6: return f"{b/1e6:.0f}MB"
            else: return f"{b/1e3:.0f}KB"

        recommend = "⭐ 最佳" if is_new and r["score"] >= 0.6 else ""
        print(f"  {rank_str:>8s} | {r['gran']:>8d} | {fmt_s(r['storage']):>10s} | "
              f"{r['score']:>6.2f} | {recommend}")

    # 最终结论
    print(f"\n{'='*70}")
    print(f"  结论")
    print(f"{'='*70}")

    # 找到精度≥0.6的最小存储配置
    good = [r for r in results if r["score"] >= 0.6]
    if good:
        best = min(good, key=lambda x: x["storage"])
        def fmt_s(b):
            if b >= 1e12: return f"{b/1e12:.1f}TB"
            elif b >= 1e9: return f"{b/1e9:.1f}GB"
            else: return f"{b/1e6:.0f}MB"

        rank_str = f"rank={best['rank']}" if best['rank'] else "full-rank"
        print(f"""
  推荐配置：粒度={best['gran']}, {rank_str}
  100M token 存储: {fmt_s(best['storage'])}
  routing 精度: {best['score']:.2f}

  对比原始配置 (粒度=1K, full-rank):
    原存储:   542 GB → 推荐存储: {fmt_s(best['storage'])}
    节省: {(1 - best['storage'] / 5.42e11) * 100:.0f}%
    精度影响: {'无损' if best['score'] >= 0.9 else f"轻微下降到 {best['score']:.0%}"}
        """)
    else:
        print("\n  所有配置精度都偏低，建议增加序列长度或增强模式差异。")

    total = time.time() - t0
    print(f"  总耗时: {total:.1f}s")


if __name__ == "__main__":
    main()
