"""
Progressive Neural Networks Baseline (Rusu et al., 2016)
每个新任务创建一个新的Transformer列，带有从旧列到新列的lateral连接。
旧列完全冻结 → 零遗忘（但参数线性增长）。
"""
import torch
import torch.nn as nn

from .base import CLModel, TransformerBlock
from ..config import ExperimentConfig


class ProgressiveBlock(nn.Module):
    """带lateral连接的Transformer块。"""

    def __init__(self, config: ExperimentConfig, num_prev_columns: int):
        super().__init__()
        self.block = TransformerBlock(config)
        # 每个之前的列提供一个lateral连接（线性投影）
        self.laterals = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(num_prev_columns)
        ])

    def forward(self, x: torch.Tensor,
                prev_features: list[torch.Tensor] = None) -> torch.Tensor:
        # 加上来自旧列同层的lateral输入
        if prev_features:
            for i, feat in enumerate(prev_features):
                if i < len(self.laterals):
                    x = x + self.laterals[i](feat)
        return self.block(x)


class ProgressiveNet(CLModel):
    """
    Progressive Neural Networks。
    每个任务拥有独立的Transformer列 + 独立的输出Head。
    旧列在新任务开始时被冻结。
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # 共享的embedding层（不冻结，用于所有列）
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.ln_f = nn.LayerNorm(config.d_model)

        # 列和Head将在on_task_start中动态添加
        # columns[i] = ModuleList of ProgressiveBlock
        self.columns = nn.ModuleList()
        # heads[i] = Linear head for task i
        self.heads = nn.ModuleList()
        self._num_columns = 0

    def on_task_start(self, task_id: int):
        """冻结所有旧列，创建新列。"""
        device = next(self.parameters()).device

        # 冻结所有现有列和Head
        for col in self.columns:
            for p in col.parameters():
                p.requires_grad = False
        for head in self.heads:
            for p in head.parameters():
                p.requires_grad = False

        # 创建新列：每个Block带有指向所有旧列的lateral连接
        new_column = nn.ModuleList([
            ProgressiveBlock(self.config, self._num_columns)
            for _ in range(self.config.n_layers)
        ]).to(device)
        self.columns.append(new_column)

        # 创建新的输出Head
        new_head = nn.Linear(self.config.d_model, self.config.vocab_size).to(device)
        self.heads.append(new_head)

        self._num_columns += 1

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h_base = (self.token_emb(x) + self.pos_emb(pos)
                  + self.task_emb(task).unsqueeze(1))

        # 确定使用哪个列（超出范围时用最后一个）
        target_col = min(task_id, self._num_columns - 1)

        # 逐列运行，收集每层的features
        # all_features[col_id][layer_id] = tensor
        all_features: list[list[torch.Tensor]] = []

        for col_id in range(target_col + 1):
            h = h_base
            col_layer_features = []

            for layer_id in range(self.config.n_layers):
                # 收集来自旧列同层的features
                prev_feats = []
                for prev_col_id in range(col_id):
                    if prev_col_id < len(all_features):
                        prev_feats.append(
                            all_features[prev_col_id][layer_id]
                        )

                h = self.columns[col_id][layer_id](h, prev_feats)
                col_layer_features.append(h)

            all_features.append(col_layer_features)

        # 使用目标列的最终输出
        out = self.ln_f(all_features[target_col][-1])
        return self.heads[target_col](out)
