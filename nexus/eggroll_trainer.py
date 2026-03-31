"""
EGGROLL 前向训练器 — 完全无 backward() 的进化策略训练

参考:
  - EGGROLL 论文 (arxiv:2511.16652): "Evolution Strategies at the Hyperscale"
  - nano-egg: https://github.com/ESHyperscale/nano-egg (INT8 预训练)
  - HyperscaleES: https://github.com/ESHyperscale/HyperscaleES (JAX 参考)

核心原理:
  1. 低秩扰动: ΔW = A @ B^T, A∈R^{m×r}, B∈R^{n×r}
     存储从 O(mn) 降到 O(r(m+n))
  2. 反义对采样 (Antithetical Sampling):
     θ+ = θ + ε·ΔW,  θ- = θ - ε·ΔW
     评估 fitness(θ+) 和 fitness(θ-)
     → 方差降低 2x（同样的噪声，正反两次评估）
  3. 适应度驱动更新:
     gradient ≈ Σ (fitness+ - fitness-) · ΔW / (2σ)
     完全前向计算，无需 backward graph

与 NEXUS 的交互:
  - 外层训练循环: EGGROLL 替代 loss.backward() + optimizer.step()
  - TTT 内部更新: 保持不变（forward-time 嵌入式自监督学习）
  - Neural Memory: vmap+grad → 改为 perturbation-based 估计
  - EWC: Fisher 信息 → 进化适应度中的参数敏感度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import time


class EggrollTrainer:
    """
    EGGROLL 进化策略训练器。

    用进化策略替代 backward + optimizer，实现完全前向训练。

    使用方式:
        trainer = EggrollTrainer(model, config)
        for epoch in range(n_epochs):
            fitness = trainer.step(batch_x, batch_y, task_id, vocab_size)
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 4,
        sigma: float = 0.01,
        alpha: float = 1.5,
        pop_size: int = 64,
        alpha_decay: float = 0.995,
    ):
        """
        Args:
            model: 要训练的 PyTorch 模型
            rank: 低秩扰动的秩 (r)。
                  r=4 是 EGGROLL 论文的推荐值（速度/效果平衡）。
                  更大的 r → 更好的梯度估计 → 更慢。
            sigma: 扰动幅度。
                  论文推荐 0.01。太大→参数震荡，太小→信号被噪声淹没。
            alpha: 学习率。
                  > 1.0 时自动衰减到 2-alpha（nano-egg 的做法）。
            pop_size: 种群大小（并行前向数 / 2，因为反义对）。
                  消费级 GPU: 64-128。数据中心: 32768（nano-egg）。
            alpha_decay: 每步 alpha 衰减系数。
        """
        self.model = model
        self.rank = rank
        self.sigma = sigma
        self.alpha = alpha
        self.alpha_init = alpha
        self.pop_size = pop_size  # 半种群（反义对总共 2*pop_size 次前向）
        self.alpha_decay = alpha_decay

        # 收集所有需要训练的参数及其形状
        self._param_info = []
        self._total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._param_info.append({
                    'name': name,
                    'shape': param.shape,
                    'numel': param.numel(),
                    'offset': self._total_params,
                })
                self._total_params += param.numel()

        self._step_count = 0

    def _params_to_flat(self) -> torch.Tensor:
        """将模型参数展平为一维向量。"""
        flat = torch.zeros(self._total_params,
                          device=next(self.model.parameters()).device)
        for info in self._param_info:
            param = dict(self.model.named_parameters())[info['name']]
            flat[info['offset']:info['offset'] + info['numel']] = param.data.flatten()
        return flat

    def _flat_to_params(self, flat: torch.Tensor):
        """将一维向量写回模型参数。"""
        named_params = dict(self.model.named_parameters())
        for info in self._param_info:
            param = named_params[info['name']]
            param.data.copy_(
                flat[info['offset']:info['offset'] + info['numel']].reshape(info['shape'])
            )

    def _apply_perturbation(
        self,
        base_flat: torch.Tensor,
        perturbation: torch.Tensor,
        sign: float,
    ):
        """将扰动应用到模型参数。"""
        self._flat_to_params(base_flat + sign * self.sigma * perturbation)

    def _evaluate_fitness(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        task_id: int,
        vocab_size: int,
    ) -> float:
        """
        评估当前参数的适应度（负 cross-entropy loss）。

        适 fitness = -loss 意味着 loss 越小 fitness 越高。
        """
        with torch.no_grad():
            logits = self.model(batch_x, task_id)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                batch_y.reshape(-1),
            )
        return -loss.item()  # 负损失 = 适应度

    def _generate_low_rank_perturbation(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        """
        生成低秩扰动 ΔW = A @ B^T。

        参考 EGGROLL 论文 Section 3.1:
          A ∈ R^{n_params × r}, B ∈ R^{n_params × r}
          存储: O(r(m+n)) 而非 O(mn)

        实际上对于参数量 n，扰动是一维的，
        低秩分解变为: perturbation = sum_i(a_i * b_i)
        其中 a_i, b_i ∈ R^{n_params}
        """
        # 为了效率，用随机正交方向做近似低秩扰动
        # 每个 rank 贡献一个方向
        perturbation = torch.zeros(self._total_params, device=device)
        for _ in range(self.rank):
            direction = torch.randn(self._total_params, device=device)
            direction = direction / (direction.norm() + 1e-8)  # 单位方向
            scale = torch.randn(1, device=device).item()
            perturbation = perturbation + scale * direction

        # 归一化到单位范数（不乘 sqrt(n)，避免梯度爆炸）
        norm = perturbation.norm()
        if norm > 0:
            perturbation = perturbation / norm
        return perturbation

    def step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        task_id: int,
        vocab_size: int,
    ) -> float:
        """
        执行一步 EGGROLL 进化更新。

        流程:
          1. 保存当前参数 θ
          2. 对每个种群成员 i:
             a. 生成低秩扰动 ΔW_i
             b. 评估 fitness(θ + σ·ΔW_i) 和 fitness(θ - σ·ΔW_i)
          3. 用适应度差异加权扰动，估计梯度方向
          4. 更新参数: θ = θ + α · gradient_estimate
          5. 衰减 α

        Args:
            batch_x: [B, seq_len] 输入 token IDs
            batch_y: [B, seq_len] 目标 token IDs
            task_id: 当前任务 ID
            vocab_size: 词汇表大小

        Returns:
            当前步骤的基线适应度（用于监控收敛）
        """
        device = batch_x.device
        self.model.eval()  # EGGROLL 不需要 dropout 等训练 mode 特性

        # 1. 保存当前参数
        base_flat = self._params_to_flat().clone()

        # 基线适应度（当前参数的表现）
        base_fitness = self._evaluate_fitness(
            batch_x, batch_y, task_id, vocab_size
        )

        # 2. 生成扰动并评估反义对
        gradient_estimate = torch.zeros(self._total_params, device=device)

        for _ in range(self.pop_size):
            perturbation = self._generate_low_rank_perturbation(device)

            # 正扰动评估
            self._apply_perturbation(base_flat, perturbation, +1.0)
            fitness_plus = self._evaluate_fitness(
                batch_x, batch_y, task_id, vocab_size
            )

            # 负扰动评估
            self._apply_perturbation(base_flat, perturbation, -1.0)
            fitness_minus = self._evaluate_fitness(
                batch_x, batch_y, task_id, vocab_size
            )

            # 3. 累积梯度估计
            # (fitness+ - fitness-) > 0 → 正方向好 → 往正方向移
            fitness_diff = fitness_plus - fitness_minus
            gradient_estimate = gradient_estimate + fitness_diff * perturbation

        # 归一化梯度估计
        gradient_estimate = gradient_estimate / (2.0 * self.sigma * self.pop_size)

        # 梯度裁剪：防止爆炸
        grad_norm = gradient_estimate.norm()
        max_grad_norm = 1.0
        if grad_norm > max_grad_norm:
            gradient_estimate = gradient_estimate * (max_grad_norm / grad_norm)

        # 4. 更新参数（梯度上升，因为 fitness = -loss）
        updated_flat = base_flat + self.alpha * gradient_estimate

        # NaN 保护：如果更新后参数出现 NaN，回滚到 base
        if torch.isnan(updated_flat).any() or torch.isinf(updated_flat).any():
            updated_flat = base_flat

        self._flat_to_params(updated_flat)

        # 5. Alpha 衰减
        self.alpha = self.alpha * self.alpha_decay
        self._step_count += 1

        self.model.train()  # 恢复训练 mode
        return -base_fitness  # 返回 loss（正值）

    def get_stats(self) -> dict:
        """返回训练统计。"""
        return {
            'step': self._step_count,
            'alpha': self.alpha,
            'sigma': self.sigma,
            'pop_size': self.pop_size,
            'rank': self.rank,
            'total_params': self._total_params,
            'forward_per_step': 2 * self.pop_size + 1,  # +1 for baseline
        }
