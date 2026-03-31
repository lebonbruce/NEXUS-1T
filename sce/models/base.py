"""
模型基础组件：
1. TransformerBlock — 标准双向Transformer块（所有方法共用）
2. CLModel — 持续学习模型的统一抽象接口
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from ..config import ExperimentConfig


class TransformerBlock(nn.Module):
    """标准双向Transformer块（Pre-LN架构）。"""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads,
            batch_first=True, dropout=config.dropout
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + 双向自注意力（无causal mask）
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class CLModel(ABC, nn.Module):
    """
    持续学习模型的统一接口。
    所有方法（Naive/EWC/Replay/Progressive/SCE）必须实现此接口。
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        前向传播。
        Args:
            x: 输入序列 (B, T) LongTensor
            task_id: 当前任务编号
        Returns:
            logits: (B, T, vocab_size)
        """
        ...

    def on_task_start(self, task_id: int):
        """任务开始前的钩子。用于结构扩展、参数冻结等。"""
        pass

    def on_task_end(self, task_id: int, train_x: torch.Tensor,
                    train_y: torch.Tensor):
        """任务结束后的钩子。用于Fisher计算、缓冲区更新等。"""
        pass

    def compute_extra_loss(self) -> torch.Tensor:
        """额外损失（EWC正则项等）。默认返回0。"""
        return torch.tensor(0.0)

    def count_params(self, trainable_only: bool = True) -> int:
        """统计参数量。"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_replay_data(self, batch_size: int):
        """获取回放数据。默认无回放。返回 None 或 (x, y, task_id)。"""
        return None
