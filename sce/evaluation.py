"""
标准持续学习评估指标。

实现三个核心CL指标：
- Average Accuracy (AA): 训练完所有任务后的平均表现
- Backward Transfer (BWT): 新任务是否导致旧任务遗忘（负=遗忘）
- Forward Transfer (FWT): 旧知识是否帮助新任务学习

参考: Lopez-Paz & Ranzato, 2017 "Gradient Episodic Memory for Continual Learning"
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_accuracy(model, X: torch.Tensor, Y: torch.Tensor,
                     task_id: int, device: torch.device,
                     batch_size: int = 256) -> float:
    """
    计算逐位置的精确匹配准确率。
    prediction[i] == target[i] 才算正确。
    """
    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = X[i:i+batch_size].to(device)
            by = Y[i:i+batch_size].to(device)
            logits = model(bx, task_id)
            preds = logits.argmax(dim=-1)  # (B, T)
            total_correct += (preds == by).sum().item()
            total_tokens += by.numel()

    model.train()
    return total_correct / total_tokens if total_tokens > 0 else 0.0


def compute_cl_metrics(accuracy_matrix: np.ndarray,
                       single_task_baselines: np.ndarray = None) -> dict:
    """
    从accuracy矩阵计算CL标准指标。

    Args:
        accuracy_matrix: shape (num_tasks, num_tasks)
            accuracy_matrix[i][j] = 训练完第j个任务后在第i个任务上的准确率
        single_task_baselines: shape (num_tasks,)
            single_task_baselines[i] = 只训练第i个任务时的准确率（用于FWT）

    Returns:
        dict with keys: AA, BWT, FWT
    """
    T = accuracy_matrix.shape[0]

    # Average Accuracy: 训练完所有任务后的平均准确率
    aa = np.mean(accuracy_matrix[:, -1])

    # Backward Transfer: 新任务对旧任务的影响
    # BWT = (1/(T-1)) * sum_{i=1}^{T-1} (a[i, T] - a[i, i])
    # 负数 = 遗忘
    if T > 1:
        bwt = np.mean([
            accuracy_matrix[i, -1] - accuracy_matrix[i, i]
            for i in range(T - 1)
        ])
    else:
        bwt = 0.0

    # Forward Transfer: 旧知识对新任务的帮助
    # FWT = (1/(T-1)) * sum_{i=2}^{T} (a[i, i] - baseline[i])
    if single_task_baselines is not None and T > 1:
        fwt = np.mean([
            accuracy_matrix[i, i] - single_task_baselines[i]
            for i in range(1, T)
        ])
    else:
        fwt = None  # 无baseline数据时不计算

    return {"AA": aa, "BWT": bwt, "FWT": fwt}
