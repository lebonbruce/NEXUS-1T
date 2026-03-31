"""
NEXUS v2 全组件开启完整测试
开启：TurboQuant QAT, MoE 分裂/合并, TTT 始终激活
关闭：NeuralMemory (vmap+grad 3-5x慢), EGGROLL (30x慢)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from sce.config import ExperimentConfig
from sce.tasks import generate_task_data, TASK_NAMES
from sce.evaluation import compute_accuracy, compute_cl_metrics
from sce.models.naive import NaiveTransformer
from nexus.config import NexusConfig
from nexus.model import NEXUSTransformer
from nexus.ewc import EWC


def main():
    print("=" * 80)
    print("  NEXUS v2 全组件开启完整测试")
    print("  ON:  TurboQuant QAT + MoE growth + TTT always")
    print("  OFF: NeuralMemory (速度杀手) + EGGROLL (toy不适用)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = NexusConfig()
    config.kv_quant_enabled = True       # TurboQuant QAT on
    config.moe_growth_enabled = True     # MoE 分裂/合并 on
    config.ttt_min_seq_len = 0           # TTT 始终激活

    print(f"  Device: {device}")
    print(f"  Tasks: {config.num_tasks} | Steps/task: {config.steps_per_task}")
    print(f"  TurboQuant: ON (K{config.kv_quant_key_bits}/V{config.kv_quant_value_bits})")
    print(f"  MoE growth: {config.moe_growth_enabled}")
    print(f"  TTT: always on (min_seq=0)")
    print(f"  NeuralMemory: off (min_seq={config.titans_min_seq_len})")
    print(f"  EGGROLL: {config.eggroll_enabled}")
    print(f"  Replay every: {config.replay_every_n_steps} steps")

    # === Naive 基线 ===
    print(f"\n{'=' * 80}")
    print("  [1/2] Naive baseline...")
    base = ExperimentConfig()
    test_data = [
        generate_task_data(t, base.test_samples, base.vocab_size, base.seq_len)
        for t in range(base.num_tasks)
    ]

    torch.manual_seed(42)
    np.random.seed(42)
    naive_model = NaiveTransformer(base).to(device)
    naive_matrix = np.zeros((base.num_tasks, base.num_tasks))
    t0 = time.time()

    for tid in range(base.num_tasks):
        tx, ty = generate_task_data(tid, base.train_samples, base.vocab_size, base.seq_len)
        naive_model.on_task_start(tid)
        opt = optim.Adam(
            [p for p in naive_model.parameters() if p.requires_grad], lr=base.lr
        )
        naive_model.train()
        for s in range(base.steps_per_task):
            idx = torch.randint(0, len(tx), (base.batch_size,))
            logits = naive_model(tx[idx].to(device), tid)
            loss = F.cross_entropy(
                logits.reshape(-1, base.vocab_size), ty[idx].to(device).reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        naive_model.on_task_end(tid, tx, ty)
        for eid in range(base.num_tasks):
            ex, ey = test_data[eid]
            naive_matrix[eid, tid] = compute_accuracy(naive_model, ex, ey, eid, device)

    naive_time = time.time() - t0
    print(f"  Naive done in {naive_time:.1f}s")

    # === NEXUS 全组件 ===
    print(f"\n{'=' * 80}")
    print("  [2/2] NEXUS (all feasible components ON)...")

    test_data2 = [
        generate_task_data(t, config.test_samples, config.vocab_size, config.seq_len)
        for t in range(config.num_tasks)
    ]

    torch.manual_seed(42)
    np.random.seed(42)
    model = NEXUSTransformer(config).to(device)
    nexus_matrix = np.zeros((config.num_tasks, config.num_tasks))

    ewc = EWC(ewc_lambda=config.ewc_lambda, exclude_patterns=["ffn.experts"])

    t0 = time.time()
    total_p = model.count_params(False)
    train_p = model.count_params(True)
    print(f"  Params: {total_p:,} total | {train_p:,} trainable")

    for tid in range(config.num_tasks):
        print(f"  --- Task {tid}: {TASK_NAMES[tid]} ---")
        tx, ty = generate_task_data(
            tid, config.train_samples, config.vocab_size, config.seq_len
        )
        model.on_task_start(tid)

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = optim.Adam(trainable, lr=config.lr)
        model.train()

        for s in range(config.steps_per_task):
            idx = torch.randint(0, len(tx), (config.batch_size,))
            bx = tx[idx].to(device)
            by = ty[idx].to(device)

            logits, kd_loss = model.forward_with_kd(bx, tid)
            task_loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size), by.reshape(-1)
            )
            loss = task_loss + kd_loss + ewc.penalty(model)

            model.check_and_grow(task_loss.item())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.grad_clip_norm)
            opt.step()

            # Replay（降频）
            if s % config.replay_every_n_steps == 0:
                replay = model.get_replay_data(config.batch_size)
                if replay is not None:
                    rx, ry, rt = replay
                    rl = model(rx, rt)
                    r_loss = F.cross_entropy(
                        rl.reshape(-1, config.vocab_size), ry.reshape(-1)
                    )
                    opt.zero_grad()
                    r_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainable, max_norm=config.grad_clip_norm
                    )
                    opt.step()

        model.on_task_end(tid, tx, ty)

        ewc.compute_fisher(
            model, tx, ty, tid,
            vocab_size=config.vocab_size,
            num_samples=config.ewc_num_samples,
            batch_size=config.batch_size,
        )

        for eid in range(config.num_tasks):
            ex, ey = test_data2[eid]
            acc = compute_accuracy(model, ex, ey, eid, device)
            nexus_matrix[eid, tid] = acc
            if eid <= tid:
                print(f"    [{TASK_NAMES[eid]:15s}]: {acc:.4f}")

    nexus_time = time.time() - t0

    # === 结果 ===
    nm = compute_cl_metrics(naive_matrix)
    xm = compute_cl_metrics(nexus_matrix)
    ratio = nexus_time / naive_time

    print(f"\n\n{'=' * 80}")
    print("  完整 Benchmark 结果（5 任务 x 500 步）")
    print(f"{'=' * 80}")

    header = f"  {'Method':<28s} | {'AA':>6s} | {'BWT':>7s} | {'Params':>8s} | {'Time':>6s} | {'Ratio':>6s}"
    print(f"\n{header}")
    print(f"  {'-' * 75}")

    naive_params = naive_model.count_params(False)
    print(f"  {'Naive':<28s} | {nm['AA']*100:5.1f}% | "
          f"{nm['BWT']*100:+6.1f}% | {naive_params/1e3:6.0f}K | "
          f"{naive_time:5.0f}s | 1.0x")
    print(f"  {'NEXUS (all components)':<28s} | {xm['AA']*100:5.1f}% | "
          f"{xm['BWT']*100:+6.1f}% | {total_p/1e3:6.0f}K | "
          f"{nexus_time:5.0f}s | {ratio:.1f}x")

    print(f"\n  Per-Task Final Accuracy:")
    print(f"  {'':28s} |", end="")
    for tn in TASK_NAMES:
        print(f" {tn[:8]:>8s}", end="")
    print()

    print(f"  {'Naive':<28s} |", end="")
    for v in naive_matrix[:, -1]:
        print(f" {v*100:7.1f}%", end="")
    print()

    print(f"  {'NEXUS (all components)':<28s} |", end="")
    for v in nexus_matrix[:, -1]:
        print(f" {v*100:7.1f}%", end="")
    print()

    # 判定
    print(f"\n  {'=' * 60}")
    speed_ok = "✅" if ratio < 10 else "❌"
    aa_ok = "✅" if xm["AA"] > nm["AA"] else "❌"
    bwt_ok = "✅" if xm["BWT"] > nm["BWT"] else "❌"
    print(f"  速度: {ratio:.1f}x (目标 <10x) {speed_ok}")
    print(f"  AA:  NEXUS {xm['AA']*100:.1f}% vs Naive {nm['AA']*100:.1f}% {aa_ok}")
    print(f"  BWT: NEXUS {xm['BWT']*100:+.1f}% vs Naive {nm['BWT']*100:+.1f}% {bwt_ok}")


if __name__ == "__main__":
    main()
