"""
EWC Baseline: Elastic Weight Consolidation (Kirkpatrick et al., 2017)
通过Fisher信息矩阵对角线约束参数不要偏离旧任务的最优解太远。
使用Online EWC变体（累积Fisher，而非为每个任务单独维护）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .base import CLModel, TransformerBlock
from ..config import ExperimentConfig


class EWCTransformer(CLModel):
    """
    Online EWC基线。
    架构与Naive完全相同，只增加Fisher正则项。
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

        # EWC状态：累积Fisher对角线 + 参数快照
        self._fisher_diag: dict[str, torch.Tensor] = {}
        self._saved_params: dict[str, torch.Tensor] = {}
        self._tasks_seen = 0

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

    def on_task_end(self, task_id: int, train_x: torch.Tensor,
                    train_y: torch.Tensor):
        """
        任务结束后计算Fisher信息对角线，并累积到全局Fisher中。
        使用Online EWC：新Fisher = gamma * 旧Fisher + 新Fisher
        """
        device = next(self.parameters()).device

        # 采样一个子集计算Fisher
        n = min(self.config.ewc_fisher_samples, len(train_x))
        idx = torch.randperm(len(train_x))[:n]
        sample_x = train_x[idx].to(device)
        sample_y = train_y[idx].to(device)

        # 计算梯度平方的期望（Fisher对角线近似）
        new_fisher: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            new_fisher[name] = torch.zeros_like(param)

        self.eval()
        # 逐样本计算梯度（更准确的Fisher估计）
        for i in range(n):
            self.zero_grad()
            logits = self(sample_x[i:i+1], task_id)
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                sample_y[i:i+1].reshape(-1)
            )
            loss.backward()
            for name, param in self.named_parameters():
                if param.grad is not None:
                    new_fisher[name] += param.grad.data ** 2

        # 归一化
        for name in new_fisher:
            new_fisher[name] /= n

        # Online累积：Fisher = 0.5 * old_Fisher + new_Fisher
        gamma = 0.5
        for name in new_fisher:
            if name in self._fisher_diag:
                self._fisher_diag[name] = (
                    gamma * self._fisher_diag[name] + new_fisher[name]
                )
            else:
                self._fisher_diag[name] = new_fisher[name]

        # 保存当前参数快照
        self._saved_params = {
            name: param.data.clone()
            for name, param in self.named_parameters()
        }
        self._tasks_seen += 1
        self.train()

    def compute_extra_loss(self) -> torch.Tensor:
        """计算EWC正则项: lambda * sum(F_i * (theta - theta_star)^2)"""
        if not self._fisher_diag:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        ewc_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, param in self.named_parameters():
            if name in self._fisher_diag:
                ewc_loss = ewc_loss + (
                    self._fisher_diag[name] *
                    (param - self._saved_params[name]) ** 2
                ).sum()

        return self.config.ewc_lambda * ewc_loss
