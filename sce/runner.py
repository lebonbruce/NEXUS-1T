"""
实验运行器：统一的CL训练和评估流程。
所有方法通过相同的训练循环运行，确保公平比较。
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from .config import ExperimentConfig
from .tasks import generate_task_data, TASK_NAMES
from .evaluation import compute_accuracy, compute_cl_metrics
from .models.sce_model import SCETransformer


def _train_one_task(model, train_x, train_y, task_id, config, device):
    """训练一个任务。所有方法共用此训练循环。"""
    # 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print(f"    警告: 没有可训练参数，跳过任务 {task_id} 的训练")
        return

    optimizer = optim.Adam(trainable_params, lr=config.lr)

    model.train()
    n = len(train_x)

    for step in range(config.steps_per_task):
        # 采样一个mini-batch
        idx = torch.randint(0, n, (config.batch_size,))
        bx = train_x[idx].to(device)
        by = train_y[idx].to(device)

        # 前向传播
        if isinstance(model, SCETransformer):
            logits, kd_loss = model.forward_with_kd(bx, task_id)
            task_loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size), by.reshape(-1)
            )
            loss = task_loss + kd_loss
            # Surprise检测和自动生长
            model.check_and_grow(task_loss.item())
        else:
            logits = model(bx, task_id)
            task_loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size), by.reshape(-1)
            )
            # 额外损失（EWC正则项等）
            extra = model.compute_extra_loss()
            loss = task_loss + extra

        # Replay: 混合旧任务数据
        replay = model.get_replay_data(config.batch_size // 2)
        if replay is not None:
            rx, ry, rt = replay
            r_logits = model(rx, rt)
            replay_loss = F.cross_entropy(
                r_logits.reshape(-1, config.vocab_size), ry.reshape(-1)
            )
            loss = loss + replay_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 进度日志
        if step % 100 == 0:
            print(f"    Step {step:3d}/{config.steps_per_task} | "
                  f"Loss: {task_loss.item():.4f}")


def run_single_experiment(config: ExperimentConfig, model_class,
                          device: torch.device) -> dict:
    """
    运行一个完整的CL实验：
    1. 依次训练5个任务
    2. 每训练完一个任务，评估所有任务的准确率
    3. 返回accuracy矩阵和参数统计

    Returns:
        dict with keys:
        - accuracy_matrix: (num_tasks, num_tasks) ndarray
        - final_params_trainable: int
        - final_params_total: int
        - duration: float (seconds)
    """
    # 预生成所有任务的测试数据
    test_data = []
    for tid in range(config.num_tasks):
        tx, ty = generate_task_data(tid, config.test_samples,
                                     config.vocab_size, config.seq_len)
        test_data.append((tx, ty))

    # 初始化模型
    model = model_class(config).to(device)
    accuracy_matrix = np.zeros((config.num_tasks, config.num_tasks))

    start_time = time.time()

    for task_id in range(config.num_tasks):
        print(f"\n  --- Task {task_id}: {TASK_NAMES[task_id]} ---")

        # 生成训练数据
        train_x, train_y = generate_task_data(
            task_id, config.train_samples,
            config.vocab_size, config.seq_len
        )

        # 前置钩子（结构扩展等）
        model.on_task_start(task_id)
        print(f"    Trainable params: {model.count_params(True):,} | "
              f"Total params: {model.count_params(False):,}")

        # 训练
        _train_one_task(model, train_x, train_y, task_id, config, device)

        # 后置钩子（Fisher计算、buffer更新等）
        model.on_task_end(task_id, train_x, train_y)

        # 评估所有任务
        for eval_id in range(config.num_tasks):
            tx, ty = test_data[eval_id]
            acc = compute_accuracy(model, tx, ty, eval_id, device)
            accuracy_matrix[eval_id, task_id] = acc
            status = "SEEN" if eval_id <= task_id else "unseen"
            if eval_id <= task_id:
                print(f"    Eval [{status}] {TASK_NAMES[eval_id]:15s}: {acc:.4f}")

    duration = time.time() - start_time

    return {
        "accuracy_matrix": accuracy_matrix,
        "final_params_trainable": model.count_params(True),
        "final_params_total": model.count_params(False),
        "duration": duration,
    }


def run_all_experiments(config: ExperimentConfig) -> dict:
    """
    运行所有方法的完整对比实验。
    每个方法运行 config.num_runs 次，报告均值和标准差。

    Returns:
        dict mapping method_name → {
            "acc_matrices": list of ndarrays,
            "metrics_mean": dict, "metrics_std": dict,
            "params_trainable": int, "params_total": int,
            "duration_mean": float,
        }
    """
    from .models.naive import NaiveTransformer
    from .models.ewc import EWCTransformer
    from .models.replay import ReplayTransformer
    from .models.progressive import ProgressiveNet
    from .models.sce_model import SCETransformer

    methods = {
        "Naive":       NaiveTransformer,
        "EWC":         EWCTransformer,
        "Replay":      ReplayTransformer,
        "Progressive": ProgressiveNet,
        "SCE (ours)":  SCETransformer,
    }

    device = torch.device(config.device)
    all_results = {}

    for name, model_class in methods.items():
        print(f"\n{'='*60}")
        print(f"Method: {name}")
        print(f"{'='*60}")

        run_metrics = []
        run_matrices = []
        run_durations = []
        run_params_t = 0
        run_params_all = 0

        for run_id in range(config.num_runs):
            print(f"\n  Run {run_id+1}/{config.num_runs}")
            # 固定种子以保证可复现性
            torch.manual_seed(config.seed + run_id)
            np.random.seed(config.seed + run_id)

            result = run_single_experiment(config, model_class, device)
            metrics = compute_cl_metrics(result["accuracy_matrix"])

            run_matrices.append(result["accuracy_matrix"])
            run_metrics.append(metrics)
            run_durations.append(result["duration"])
            run_params_t = result["final_params_trainable"]
            run_params_all = result["final_params_total"]

        # 聚合统计
        aa_vals = [m["AA"] for m in run_metrics]
        bwt_vals = [m["BWT"] for m in run_metrics]

        all_results[name] = {
            "acc_matrices": run_matrices,
            "metrics_mean": {
                "AA": np.mean(aa_vals),
                "BWT": np.mean(bwt_vals),
            },
            "metrics_std": {
                "AA": np.std(aa_vals),
                "BWT": np.std(bwt_vals),
            },
            "params_trainable": run_params_t,
            "params_total": run_params_all,
            "duration_mean": np.mean(run_durations),
        }

    return all_results
