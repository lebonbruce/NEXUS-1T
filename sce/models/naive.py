"""
Naive Baseline: 标准Transformer直接微调。
这是灾难性遗忘的"上界"——新任务训练会完全覆写旧知识。
"""
import torch
import torch.nn as nn

from .base import CLModel, TransformerBlock
from ..config import ExperimentConfig


class NaiveTransformer(CLModel):
    """
    朴素微调基线。
    架构：Token Embedding + Position Embedding + Task Embedding
          + 4层双向Transformer + Output Head
    无任何抗遗忘机制。
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

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)

        # 三种embedding叠加
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)

        for block in self.blocks:
            h = block(h)

        return self.head(self.ln_f(h))
