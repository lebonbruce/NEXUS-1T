"""
NEXUS v2 快速验证脚本 — 审计优化后的 Smoke Test

目标：< 5 分钟完成，验证：
  1. 优化后的 NEXUS 能正常前向/反向（不报错）
  2. AA/BWT 指标没有 regression
  3. 训练速度提升（对比 Naive 基线的倍数）

配置：
  - steps_per_task: 100（从 500 降到 100）
  - num_tasks: 3（从 5 降到 3，足以测 BWT）
  - train_samples: 2000（从 5000 降到 2000）
"""
import sys
import os
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


def train_nexus(model, train_x, train_y, task_id, config, device, ewc=None):
    """优化后的 NEXUS backward 训练。"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return

    optimizer = optim.Adam(trainable_params, lr=config.lr)
    model.train()
    V = config.vocab_size
    n = len(train_x)
    replay_every = config.replay_every_n_steps

    for step in range(config.steps_per_task):
        idx = torch.randint(0, n, (config.batch_size,))
        bx = train_x[idx].to(device)
        by = train_y[idx].to(device)

        logits, kd_loss = model.forward_with_kd(bx, task_id)
        task_loss = F.cross_entropy(logits.reshape(-1, V), by.reshape(-1))
        loss = task_loss + kd_loss

        if ewc is not None:
            loss = loss + ewc.penalty(model)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip_norm)
        optimizer.step()

        # Replay（降频）
        if step % replay_every == 0:
            replay = model.get_replay_data(config.batch_size)
            if replay is not None:
                rx, ry, r_task = replay
                r_logits = model(rx, r_task)
                r_loss = F.cross_entropy(r_logits.reshape(-1, V), ry.reshape(-1))
                optimizer.zero_grad()
                r_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip_norm)
                optimizer.step()


def run_naive(config, device, num_tasks):
    """Naive 基线。"""
    base_config = ExperimentConfig()
    base_config.steps_per_task = config.steps_per_task
    base_config.train_samples = config.train_samples
    base_config.test_samples = config.test_samples

    test_data = [
        generate_task_data(tid, base_config.test_samples,
                          base_config.vocab_size, base_config.seq_len)
        for tid in range(num_tasks)
    ]

    model = NaiveTransformer(base_config).to(device)
    accuracy_matrix = np.zeros((num_tasks, num_tasks))

    start_time = time.time()

    for task_id in range(num_tasks):
        train_x, train_y = generate_task_data(
            task_id, base_config.train_samples,
            base_config.vocab_size, base_config.seq_len
        )
        model.on_task_start(task_id)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr=base_config.lr)
        model.train()

        for step in range(base_config.steps_per_task):
            idx = torch.randint(0, len(train_x), (base_config.batch_size,))
            bx = train_x[idx].to(device)
            by = train_y[idx].to(device)
            logits = model(bx, task_id)
            loss = F.cross_entropy(
                logits.reshape(-1, base_config.vocab_size), by.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.on_task_end(task_id, train_x, train_y)

        for eval_id in range(num_tasks):
            tx, ty = test_data[eval_id]
            acc = compute_accuracy(model, tx, ty, eval_id, device)
            accuracy_matrix[eval_id, task_id] = acc

    duration = time.time() - start_time
    return accuracy_matrix, duration, model.count_params(False)


def run_nexus_optimized(config, device, num_tasks):
    """优化后的 NEXUS。"""
    test_data = [
        generate_task_data(tid, config.test_samples,
                          config.vocab_size, config.seq_len)
        for tid in range(num_tasks)
    ]

    model = NEXUSTransformer(config).to(device)
    accuracy_matrix = np.zeros((num_tasks, num_tasks))

    ewc = EWC(
        ewc_lambda=config.ewc_lambda,
        exclude_patterns=['ffn.experts'],
    )

    start_time = time.time()

    total_params = model.count_params(False)
    trainable_params = model.count_params(True)
    print(f"  NEXUS params: {total_params:,} total | {trainable_params:,} trainable")

    for task_id in range(num_tasks):
        print(f"  --- Task {task_id}: {TASK_NAMES[task_id]} ---")

        train_x, train_y = generate_task_data(
            task_id, config.train_samples, config.vocab_size, config.seq_len
        )
        model.on_task_start(task_id)

        train_nexus(model, train_x, train_y, task_id, config, device, ewc)

        model.on_task_end(task_id, train_x, train_y)

        ewc.compute_fisher(
            model, train_x, train_y, task_id,
            vocab_size=config.vocab_size,
            num_samples=20,
            batch_size=config.batch_size,
        )

        for eval_id in range(num_tasks):
            tx, ty = test_data[eval_id]
            acc = compute_accuracy(model, tx, ty, eval_id, device)
            accuracy_matrix[eval_id, task_id] = acc
            if eval_id <= task_id:
                print(f"    [{TASK_NAMES[eval_id]:15s}]: {acc:.4f}")

    duration = time.time() - start_time
    return accuracy_matrix, duration, total_params


def main():
    print("=" * 70)
    print("  NEXUS v2 快速验证 — 审计优化后 Smoke Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = 3  # 3 个任务足以验证 BWT

    # 配置：快速验证参数
    config = NexusConfig()
    config.steps_per_task = 100
    config.train_samples = 2000
    config.test_samples = 500
    config.ewc_num_samples = 20

    print(f"  Device: {device}")
    print(f"  Tasks: {num_tasks} | Steps/task: {config.steps_per_task}")
    print(f"  seq_len: {config.seq_len} | d_model: {config.d_model}")
    print(f"  TTT min_seq: {config.ttt_min_seq_len} | Titans min_seq: {config.titans_min_seq_len}")
    print(f"  MoE growth: {config.moe_growth_enabled} | Replay every: {config.replay_every_n_steps}")
    print(f"  EGGROLL: {config.eggroll_enabled}")

    # 1. Naive 基线
    print(f"\n{'=' * 70}")
    print("  [1/2] Running Naive baseline...")
    print(f"{'=' * 70}")
    torch.manual_seed(42)
    np.random.seed(42)
    naive_matrix, naive_time, naive_params = run_naive(config, device, num_tasks)

    # 2. NEXUS 优化版
    print(f"\n{'=' * 70}")
    print("  [2/2] Running NEXUS (optimized)...")
    print(f"{'=' * 70}")
    torch.manual_seed(42)
    np.random.seed(42)
    nexus_matrix, nexus_time, nexus_params = run_nexus_optimized(config, device, num_tasks)

    # 结果对比
    naive_metrics = compute_cl_metrics(naive_matrix)
    nexus_metrics = compute_cl_metrics(nexus_matrix)

    speed_ratio = nexus_time / naive_time

    print(f"\n\n{'=' * 70}")
    print("  快速验证结果")
    print(f"{'=' * 70}")

    print(f"\n  {'Method':<22s} | {'AA':>6s} | {'BWT':>7s} | {'Time':>6s} | {'Ratio':>6s}")
    print(f"  {'-'*60}")
    print(f"  {'Naive':<22s} | {naive_metrics['AA']*100:5.1f}% | "
          f"{naive_metrics['BWT']*100:+6.1f}% | {naive_time:5.1f}s | 1.0x")
    print(f"  {'NEXUS (optimized)':<22s} | {nexus_metrics['AA']*100:5.1f}% | "
          f"{nexus_metrics['BWT']*100:+6.1f}% | {nexus_time:5.1f}s | {speed_ratio:.1f}x")

    print(f"\n  Per-Task ({num_tasks} tasks):")
    print(f"  {'':22s} |", end="")
    for i in range(num_tasks):
        print(f" {TASK_NAMES[i][:8]:>8s}", end="")
    print()

    print(f"  {'Naive':<22s} |", end="")
    for v in naive_matrix[:num_tasks, -1]:
        print(f" {v*100:7.1f}%", end="")
    print()

    print(f"  {'NEXUS (optimized)':<22s} |", end="")
    for v in nexus_matrix[:num_tasks, -1]:
        print(f" {v*100:7.1f}%", end="")
    print()

    # 验证判定
    print(f"\n  {'=' * 60}")
    print("  验证判定：")

    # 速度目标
    if speed_ratio < 10:
        print(f"  ✅ 速度: {speed_ratio:.1f}x（目标 <10x）— 达标")
    else:
        print(f"  ❌ 速度: {speed_ratio:.1f}x（目标 <10x）— 未达标")

    # AA 目标
    if nexus_metrics['AA'] > naive_metrics['AA']:
        print(f"  ✅ AA: NEXUS {nexus_metrics['AA']*100:.1f}% > Naive {naive_metrics['AA']*100:.1f}%")
    else:
        print(f"  ⚠️  AA: NEXUS {nexus_metrics['AA']*100:.1f}% <= Naive {naive_metrics['AA']*100:.1f}%")

    # BWT 目标
    if nexus_metrics['BWT'] > naive_metrics['BWT']:
        print(f"  ✅ BWT: NEXUS {nexus_metrics['BWT']*100:+.1f}% > Naive {naive_metrics['BWT']*100:+.1f}%")
    else:
        print(f"  ⚠️  BWT: NEXUS {nexus_metrics['BWT']*100:+.1f}% <= Naive {naive_metrics['BWT']*100:+.1f}%")


if __name__ == "__main__":
    main()
