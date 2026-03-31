"""
NEXUS Benchmark — 精简版

只对比 2 种方法:
  1. Naive — 标准 Transformer（无任何持续学习机制，传统基线）
  2. NEXUS — 综合前沿技术的新架构

评估指标:
  - AA (Average Accuracy): 训练完所有任务后的平均表现
  - BWT (Backward Transfer): 新任务是否导致旧任务遗忘（负=遗忘）
  - 参数效率: 每1K参数贡献的 AA 值

用法:
  python nexus_benchmark.py
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


def _train_one_task(model, train_x, train_y, task_id, config, device, ewc=None):
    """
    训练一个任务。
    NEXUS 使用 forward_with_kd + 独立 Replay + EWC 约束，Naive 使用标准 forward。

    v3 改动：
    1. 添加 EWC 弹性权重巩固（防止旧任务重要参数偏移）
    2. LoRA adapter 替代完整 expert 深拷贝
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print(f"    警告: 没有可训练参数，跳过任务 {task_id} 的训练")
        return

    optimizer = optim.Adam(trainable_params, lr=config.lr)
    model.train()
    n = len(train_x)

    is_nexus = isinstance(model, NEXUSTransformer)
    V = config.vocab_size

    for step in range(config.steps_per_task):
        idx = torch.randint(0, n, (config.batch_size,))
        bx = train_x[idx].to(device)
        by = train_y[idx].to(device)

        if is_nexus:
            logits, kd_loss = model.forward_with_kd(bx, task_id)
            task_loss = F.cross_entropy(
                logits.reshape(-1, V), by.reshape(-1)
            )
            loss = task_loss + kd_loss

            # === EWC 约束：防止重要参数偏移 ===
            if ewc is not None:
                ewc_loss = ewc.penalty(model)
                loss = loss + ewc_loss

            model.check_and_grow(task_loss.item())
        else:
            logits = model(bx, task_id)
            task_loss = F.cross_entropy(
                logits.reshape(-1, V), by.reshape(-1)
            )
            extra = model.compute_extra_loss()
            loss = task_loss + extra

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（使用配置值）
        clip_norm = getattr(config, 'grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)

        optimizer.step()

        # === 独立 Replay（每步执行）===
        replay = model.get_replay_data(config.batch_size)
        if replay is not None:
            rx, ry, r_task = replay
            r_logits = model(rx, r_task)
            r_loss = F.cross_entropy(
                r_logits.reshape(-1, V), ry.reshape(-1)
            )
            optimizer.zero_grad()
            r_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)
            optimizer.step()

        if step % 100 == 0:
            print(f"    Step {step:3d}/{config.steps_per_task} | "
                  f"Loss: {task_loss.item():.4f}")


def run_single_experiment(config, model_class, device):
    """运行单次实验。"""
    # 预生成测试数据
    test_data = []
    for tid in range(config.num_tasks):
        tx, ty = generate_task_data(tid, config.test_samples,
                                     config.vocab_size, config.seq_len)
        test_data.append((tx, ty))

    # 初始化模型
    model = model_class(config).to(device)
    accuracy_matrix = np.zeros((config.num_tasks, config.num_tasks))

    # EWC 实例（仅 NEXUS 使用）
    # 排除 FFN 参数：shared expert 需要自由演化适应所有任务，
    # LoRA adapters 已由冻结机制保护。EWC 只保护共享层（DiffAttn/TTT/Memory）
    ewc = None
    is_nexus = (model_class == NEXUSTransformer)
    if is_nexus:
        ewc = EWC(
            ewc_lambda=config.ewc_lambda,
            exclude_patterns=['ffn.experts'],
        )

    start_time = time.time()

    for task_id in range(config.num_tasks):
        print(f"\n  --- Task {task_id}: {TASK_NAMES[task_id]} ---")

        train_x, train_y = generate_task_data(
            task_id, config.train_samples,
            config.vocab_size, config.seq_len
        )

        model.on_task_start(task_id)
        print(f"    Trainable params: {model.count_params(True):,} | "
              f"Total params: {model.count_params(False):,}")

        _train_one_task(model, train_x, train_y, task_id, config, device, ewc=ewc)

        # 任务结束：计算 Fisher + 保存 Replay
        model.on_task_end(task_id, train_x, train_y)
        if ewc is not None:
            ewc.compute_fisher(
                model, train_x, train_y, task_id,
                vocab_size=config.vocab_size,
                num_samples=getattr(config, 'ewc_num_samples', 100),
                batch_size=config.batch_size,
            )

        for eval_id in range(config.num_tasks):
            tx, ty = test_data[eval_id]
            acc = compute_accuracy(model, tx, ty, eval_id, device)
            accuracy_matrix[eval_id, task_id] = acc
            if eval_id <= task_id:
                print(f"    Eval [{TASK_NAMES[eval_id]:15s}]: {acc:.4f}")

    duration = time.time() - start_time

    return {
        "accuracy_matrix": accuracy_matrix,
        "final_params_trainable": model.count_params(True),
        "final_params_total": model.count_params(False),
        "duration": duration,
    }


def run_benchmark():
    """运行 NEXUS vs Naive 对比实验。"""
    nexus_config = NexusConfig()
    base_config = ExperimentConfig()

    device = torch.device(nexus_config.device)
    print(f"Device: {device}")
    print(f"Config: d_model={base_config.d_model}, n_layers={base_config.n_layers}, "
          f"seq_len={base_config.seq_len}, vocab={base_config.vocab_size}, "
          f"steps_per_task={base_config.steps_per_task}")

    # 只测两个方法
    methods = {
        "Naive":  (NaiveTransformer, base_config),
        "NEXUS":  (NEXUSTransformer, nexus_config),
    }

    all_results = {}

    for name, (model_class, config) in methods.items():
        print(f"\n{'='*60}")
        print(f"Method: {name}")
        print(f"{'='*60}")

        run_metrics = []
        run_matrices = []

        for run_id in range(config.num_runs):
            print(f"\n  Run {run_id+1}/{config.num_runs}")
            torch.manual_seed(config.seed + run_id)
            np.random.seed(config.seed + run_id)

            result = run_single_experiment(config, model_class, device)
            metrics = compute_cl_metrics(result["accuracy_matrix"])

            run_matrices.append(result["accuracy_matrix"])
            run_metrics.append(metrics)

        aa_vals = [m["AA"] for m in run_metrics]
        bwt_vals = [m["BWT"] for m in run_metrics]

        all_results[name] = {
            "AA_mean": np.mean(aa_vals),
            "AA_std": np.std(aa_vals),
            "BWT_mean": np.mean(bwt_vals),
            "BWT_std": np.std(bwt_vals),
            "params_trainable": result["final_params_trainable"],
            "params_total": result["final_params_total"],
            "duration": result["duration"],
            "matrices": run_matrices,
        }

    # === 结果汇总 ===
    print(f"\n\n{'='*80}")
    print("NEXUS vs Naive Transformer — Benchmark Results")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'AA (mean±std)':>15} {'BWT (mean±std)':>17} "
          f"{'Params(train)':>13} {'Params(total)':>13} {'Time(s)':>8}")
    print("-" * 80)

    for name, r in all_results.items():
        print(f"{name:<15} "
              f"{r['AA_mean']*100:6.2f}±{r['AA_std']*100:.2f}%     "
              f"{r['BWT_mean']*100:+6.2f}±{r['BWT_std']*100:.2f}%     "
              f"{r['params_trainable']:>10,}   "
              f"{r['params_total']:>10,}   "
              f"{r['duration']:>7.1f}")

    # === 效率对比 ===
    print(f"\n{'='*80}")
    print("PARAMETER EFFICIENCY (AA per 1K trainable params)")
    print(f"{'='*80}")
    for name, r in all_results.items():
        efficiency = r["AA_mean"] * 1000 / max(r["params_trainable"], 1)
        print(f"  {name:<15}: {efficiency:.6f}")

    # === Accuracy Matrices ===
    print(f"\n{'='*80}")
    print("ACCURACY MATRICES (last run)")
    print(f"{'='*80}")
    for name, r in all_results.items():
        print(f"\n{name}:")
        print(np.array2string(r["matrices"][-1], precision=4, suppress_small=True))

    return all_results


if __name__ == "__main__":
    results = run_benchmark()
