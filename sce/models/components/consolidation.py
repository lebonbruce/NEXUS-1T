"""
Knowledge Distillation Consolidation: 新Expert继承旧Expert的泛化能力。

这是V21完全缺失的关键环节。V21的冻结策略保证零遗忘但不允许知识迁移。
通过蒸馏，新Expert在学习新任务的同时，不会完全丢弃旧Expert已学到的通用特征。
"""
import torch
import torch.nn.functional as F


def compute_kd_loss(logits_new: torch.Tensor, logits_old: torch.Tensor,
                    temperature: float = 2.0) -> torch.Tensor:
    """
    计算Knowledge Distillation损失。

    使用soft target distribution进行蒸馏：
    loss = KL( softmax(new/T) || softmax(old/T) ) * T^2

    Args:
        logits_new: 新Expert的logits (B, T, V)
        logits_old: 旧Expert的logits (B, T, V)，已detach
        temperature: 软化温度（越高越平滑，传递越多"暗知识"）

    Returns:
        kd_loss: 标量
    """
    # 在vocab维度上做softmax
    p_new = F.log_softmax(logits_new / temperature, dim=-1)
    p_old = F.softmax(logits_old / temperature, dim=-1)

    # KL散度 * T^2（标准KD缩放因子）
    kd_loss = F.kl_div(p_new, p_old, reduction="batchmean") * (temperature ** 2)

    return kd_loss
