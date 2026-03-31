"""
Elastic Weight Consolidation (EWC) — 弹性权重巩固

数学原理：
    L_total = L_task + (λ/2) · Σ_i F_i · (θ_i - θ*_i)²

其中：
    F_i = Fisher Information Matrix 对角近似（参数 i 对旧任务的重要性）
    θ*_i = 旧任务训练完成后的最优参数值
    λ = 正则化强度

Fisher Information 的直觉：
    - F_i 大 → 参数 i 对旧任务很重要 → 移动成本高 → 保持不变
    - F_i 小 → 参数 i 对旧任务不重要 → 可以自由调整 → 适应新任务

参考：Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """
    在线 EWC（Online EWC）实现。

    与标准 EWC 的区别：
    - 标准 EWC 为每个旧任务维护独立的 Fisher + 参数快照
    - 在线 EWC 使用指数滑动平均合并多任务的 Fisher
    - 内存 O(|θ|) 而非 O(T·|θ|)，更适合长序列持续学习
    """

    def __init__(self, ewc_lambda: float = 400.0, exclude_patterns: list[str] = None):
        """
        Args:
            ewc_lambda: 正则化强度。典型范围 100-5000。
                过小→遗忘，过大→无法学新任务。
                400 是 5-task seq2seq 场景的经验值。
            exclude_patterns: 要排除的参数名模式列表。
                匹配的参数不会被 EWC 约束，可以自由演化。
                例如: ['ffn.shared'] 排除 FFN 共享专家。
        """
        self.ewc_lambda = ewc_lambda
        self._exclude_patterns = exclude_patterns or []
        # Fisher 对角阵（在线累积）
        self._fisher: dict[str, torch.Tensor] = {}
        # 旧任务的最优参数
        self._optimal_params: dict[str, torch.Tensor] = {}
        # 已处理的任务数
        self._task_count = 0
        # Fisher 衰减系数（0.9 表示最近的任务权重更高）
        self._fisher_decay = 0.9

    def _should_protect(self, name: str) -> bool:
        """检查参数是否应被 EWC 保护（不匹配任何排除模式）。"""
        for pattern in self._exclude_patterns:
            if pattern in name:
                return False
        return True

    def compute_fisher(
        self,
        model: nn.Module,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        task_id: int,
        vocab_size: int,
        num_samples: int = 100,
        batch_size: int = 64,
    ):
        """
        任务结束后调用：计算 Fisher Information 并更新参数快照。

        Fisher 的计算采用 empirical Fisher 近似：
            F_i ≈ (1/N) Σ_n (∂L_n / ∂θ_i)²

        这是对角近似（忽略参数间的相关性），计算和存储都是 O(|θ|)。
        """
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        # 收集可训练参数名
        param_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]

        # 累积梯度平方（排除不应保护的参数）
        fisher_acc = {
            name: torch.zeros_like(p)
            for name, p in model.named_parameters()
            if p.requires_grad and self._should_protect(name)
        }

        n = len(train_x)
        for _ in range(num_samples):
            idx = torch.randint(0, n, (batch_size,))
            bx = train_x[idx].to(device)
            by = train_y[idx].to(device)

            model.zero_grad()
            logits = model(bx, task_id)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), by.reshape(-1)
            )
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and self._should_protect(name):
                    fisher_acc[name] += param.grad.detach().pow(2)

        # 平均
        for name in fisher_acc:
            fisher_acc[name] /= num_samples

        # 在线合并：指数衰减旧 Fisher + 新 Fisher
        if self._task_count == 0:
            self._fisher = fisher_acc
        else:
            decay = self._fisher_decay
            for name in fisher_acc:
                if name in self._fisher:
                    # 旧 Fisher 衰减 + 新 Fisher 叠加
                    self._fisher[name] = (
                        decay * self._fisher[name] + fisher_acc[name]
                    )
                else:
                    # 新参数（如新 expert 的参数）
                    self._fisher[name] = fisher_acc[name]

        # 快照当前最优参数（只保护的参数）
        self._optimal_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad and self._should_protect(name)
        }

        self._task_count += 1

        if was_training:
            model.train()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        计算 EWC 正则化罚项（动态 λ 版）：
            (λ / (1 + task_count)) / 2 · Σ_i F_i · (θ_i - θ*_i)²

        λ 按任务数自动衰减：
        - 任务 0 后：λ_eff = λ（正常约束）
        - 任务 3 后：λ_eff = λ/4（Fisher 已累积 4 个任务的信息，自然更强）
        这防止了累积 Fisher 导致的过约束和 NaN。
        """
        if not self._fisher:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        # 动态 λ：随任务数自动衰减
        effective_lambda = self.ewc_lambda / (1.0 + self._task_count)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._fisher:
                loss = loss + (
                    self._fisher[name] * (param - self._optimal_params[name]).pow(2)
                ).sum()

        return (effective_lambda / 2.0) * loss
