"""
Structural Cognitive Engine (SCE): 我们的核心方法。

三个机制的整合：
A) Bidirectional Delta Attention — 替代标准Attention，提供递归关联记忆
B) Surprise-Driven Expert Growth — FFN层自动检测domain shift并生长
C) Knowledge Distillation Consolidation — 新Expert继承旧Expert的泛化能力

哲学："用结构换算力" — 当新知识到来时，用物理结构隔离来避免覆写
      "用记忆换智商" — Delta Attention的快速权重提供上下文内的即时学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CLModel
from .components.delta_attention import DeltaRuleAttention
from .components.surprise_growth import SurpriseFFN, SurpriseMonitor
from .components.consolidation import compute_kd_loss
from ..config import ExperimentConfig


class SCEBlock(nn.Module):
    """SCE Transformer块：Delta Attention + Surprise FFN。"""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = DeltaRuleAttention(config.d_model, config.n_heads)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = SurpriseFFN(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        # Pre-LN + Delta Attention（共享，不冻结）
        h = self.ln1(x)
        x = x + self.attn(h)
        # Pre-LN + Surprise FFN（可能有多个Expert）
        x = x + self.ffn(self.ln2(x), task_id=task_id)
        return x


class SCETransformer(CLModel):
    """
    Structural Cognitive Engine。

    设计原则：
    - Attention层（Delta Rule）：所有任务共享，提供通用的"如何处理序列"能力
    - FFN层（Surprise Growth）：任务特化，按需生长，提供"如何计算特定任务"能力
    - Embedding + Head：所有任务共享
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)

        self.blocks = nn.ModuleList([
            SCEBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

        # Surprise监测器（全局共享）
        self.surprise_monitor = SurpriseMonitor(
            alpha=config.surprise_ema_alpha,
            sigma=config.surprise_sigma,
            warmup=config.surprise_warmup,
        )
        self._kd_alpha = config.kd_alpha
        self._kd_temp = config.kd_temperature
        self._has_old_expert = False

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)
        for block in self.blocks:
            h = block(h, task_id=task_id)
        return self.head(self.ln_f(h))

    def check_and_grow(self, loss_value: float):
        """
        让SurpriseMonitor判断是否需要生长。
        如果触发，所有FFN层同时生长。
        """
        should_grow = self.surprise_monitor.update(loss_value)
        if should_grow:
            self._do_grow()

    def _do_grow(self):
        """执行结构生长：冻结旧Expert，创建新Expert。"""
        device = next(self.parameters()).device
        for block in self.blocks:
            block.ffn.grow(device)
        self._has_old_expert = True
        num_experts = len(self.blocks[0].ffn.experts)
        print(f"    [SCE] Growing new experts. "
              f"Total experts per layer: {num_experts}")

    def on_task_start(self, task_id: int):
        """
        新任务开始：强制生长新Expert（保证每个任务有独立Expert）。
        这与Progressive的"每任务一个column"类似，但共享attention+embedding。
        """
        self.surprise_monitor.reset_trigger()
        # 第一个任务不需要生长（使用初始Expert）
        if task_id > 0:
            self._do_grow()
        # 注册当前task使用的（刚生长的）expert
        for block in self.blocks:
            block.ffn.register_task(task_id)

    def on_task_end(self, task_id: int, train_x: torch.Tensor,
                    train_y: torch.Tensor):
        """任务结束：确保映射是最新的（防止中途growth导致的不一致）。"""
        for block in self.blocks:
            block.ffn.register_task(task_id)

    def compute_extra_loss(self) -> torch.Tensor:
        """
        KD损失：如果存在旧Expert，让新Expert的FFN输出
        不要偏离旧Expert太远。
        """
        # KD在train_step中由runner通过forward_with_kd处理
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward_with_kd(self, x: torch.Tensor, task_id: int):
        """
        带KD的前向传播。
        返回: (logits, kd_loss)
        """
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)

        kd_loss = torch.tensor(0.0, device=device)
        kd_count = 0

        for block in self.blocks:
            h_ln = block.ln1(h)
            h = h + block.attn(h_ln)

            h_ffn_in = block.ln2(h)

            # 当前Expert的输出
            ffn_out = block.ffn(h_ffn_in, task_id=task_id)

            # 如果有旧Expert，计算KD
            if self._has_old_expert and self.training:
                old_out = block.ffn.get_old_expert_output(h_ffn_in)
                if old_out is not None:
                    # 在feature空间做KD（L2距离）
                    kd_loss = kd_loss + F.mse_loss(ffn_out, old_out)
                    kd_count += 1

            h = h + ffn_out

        logits = self.head(self.ln_f(h))

        if kd_count > 0:
            kd_loss = self._kd_alpha * kd_loss / kd_count

        return logits, kd_loss
