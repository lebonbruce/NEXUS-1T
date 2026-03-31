"""
Replay Baseline: Experience Replay（经验回放）。
每个旧任务保存少量训练样本，在学习新任务时混合回放，减缓遗忘。
"""
import torch
import torch.nn as nn

from .base import CLModel, TransformerBlock
from ..config import ExperimentConfig


class ReplayTransformer(CLModel):
    """
    经验回放基线。
    架构与Naive完全相同，增加一个replay buffer。
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

        # Replay buffer: 每个旧任务的 (X, Y, task_id)
        self._buffer_x: list[torch.Tensor] = []
        self._buffer_y: list[torch.Tensor] = []
        self._buffer_tasks: list[int] = []

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
        """任务结束后，从训练数据中随机采样存入buffer。"""
        n = min(self.config.replay_buffer_per_task, len(train_x))
        idx = torch.randperm(len(train_x))[:n]
        # 存储在CPU以节省显存
        self._buffer_x.append(train_x[idx].cpu())
        self._buffer_y.append(train_y[idx].cpu())
        self._buffer_tasks.append(task_id)

    def get_replay_data(self, batch_size: int):
        """
        从buffer中随机采样一个旧任务的一个mini-batch。
        Returns: (x, y, task_id) 或 None（buffer为空时）
        """
        if not self._buffer_x:
            return None

        # 随机选一个旧任务
        buf_idx = torch.randint(0, len(self._buffer_x), (1,)).item()
        bx = self._buffer_x[buf_idx]
        by = self._buffer_y[buf_idx]
        task = self._buffer_tasks[buf_idx]

        # 从中随机采样
        n = min(batch_size, len(bx))
        sample_idx = torch.randint(0, len(bx), (n,))

        device = next(self.parameters()).device
        return bx[sample_idx].to(device), by[sample_idx].to(device), task
