"""
NEXUS 神经长期记忆模块

严格参考 lucidrains/titans-pytorch 官方实现:
  https://github.com/lucidrains/titans-pytorch/blob/main/titans_pytorch/neural_memory.py

核心原理（来自 Google Titans 论文 + lucidrains 实现）：
  记忆 = 一个小 MLP 的权重 θ
  存储 = 对 MLP 做在线 SGD: loss = ||MLP(key) - value||²
  检索 = MLP(query) 的前向传播
  surprise = loss 值本身（loss高 = 信息新颖 → 更大学习率）

关键实现细节（来自源码，非论文推测）：
  1. 使用 torch.func.vmap + torch.func.grad 做 per-sample 梯度
  2. 自适应学习率: adaptive_lr = sigmoid(linear(x)) * max_lr
  3. 动量更新: 支持 1st/2nd 阶动量（通过 associative scan）
  4. 权重衰减: 防止记忆无限膨胀的遗忘门机制
  5. 存储和检索使用不同的 chunk_size
  6. store 前先分 chunk → 每 chunk 一个梯度更新

简化：
  - 去掉分布式/多GPU支持
  - 去掉 tensordict 依赖（用原生 dict）
  - 去掉 associative scan（用简单循环替代）
  - 去掉 hyper connections / multi-view
  - 保留核心: vmap grad, adaptive lr, momentum, weight decay
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad

from .config import NexusConfig


class MemoryMLP(nn.Module):
    """
    记忆网络 — 一个小 MLP，其权重就是"记忆"。
    参考: titans_pytorch/memory_models.py MemoryMLP

    Input → Linear → GELU → Linear → Output (same dim)
    """
    def __init__(self, dim: int, expansion_factor: float = 4.0):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualNorm(nn.Module):
    """
    LayerNorm + Residual wrapper for memory model.
    参考: titans_pytorch/memory_models.py ResidualNorm

    TTT 论文做法: output = LN(model(x)) + x
    """
    def __init__(self, dim: int, model: nn.Module):
        super().__init__()
        self.model = model
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.model(x)) + x


class NeuralMemory(nn.Module):
    """
    Titans 神经长期记忆。

    严格遵循 lucidrains 实现的存储/检索分离架构:
    - store_memories(): key-value 对通过在线 SGD 写入 MLP 权重
    - retrieve_memories(): query 通过 MLP 前向传播检索
    - forward(): store → retrieve 的完整流程

    简化版: 去掉多头、associative scan、hyper connections
    保留核心: vmap grad, adaptive lr, weight decay
    """

    def __init__(self, config: NexusConfig):
        super().__init__()
        self.dim = config.d_model
        self.chunk_size = config.msa_chunk_size

        # === 记忆模型 (权重就是记忆) ===
        mem_model = MemoryMLP(self.dim, expansion_factor=2.0)
        self.memory_model = ResidualNorm(dim=self.dim, model=mem_model)

        # 保存初始参数名和形状
        self.mem_param_names = []
        self.mem_param_shapes = []
        for name, p in self.memory_model.named_parameters():
            self.mem_param_names.append(name)
            self.mem_param_shapes.append(p.shape)

        # === QKV 投影 ===
        self.to_queries = nn.Linear(self.dim, self.dim, bias=False)
        self.to_keys = nn.Linear(self.dim, self.dim, bias=False)
        self.to_values = nn.Linear(self.dim, self.dim, bias=False)

        # === 自适应学习率 ===
        # 参考: neural_memory.py 第443-446行
        self.to_adaptive_step = nn.Linear(self.dim, 1)

        # === 权重衰减门 ===
        # 参考: neural_memory.py 第507-509行
        self.to_decay_factor = nn.Linear(self.dim, 1)

        # === 动量参数 ===
        # 参考: neural_memory.py 第455-458行
        self.to_momentum = nn.Linear(self.dim, 1)

        # === 归一化 ===
        self.store_norm = nn.RMSNorm(self.dim)
        self.retrieve_norm = nn.RMSNorm(self.dim)
        self.output_norm = nn.RMSNorm(self.dim)

        # 输出门控
        self.output_gate = nn.Sequential(
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )

        # === 构建 per-sample 梯度函数 (核心! 参考第392-402行) ===
        def forward_and_loss(params, inputs, loss_weights, target):
            """
            对记忆模型做前向传播并计算 MSE loss。
            参考: neural_memory.py 第392-396行
            loss = |M(k) - v|² — 论文 eq(12)
            """
            pred = functional_call(self.memory_model, params, inputs)
            loss = (pred - target).pow(2).mean(dim=-1)
            weighted_loss = loss * loss_weights
            return weighted_loss.sum(), loss

        grad_fn = grad(forward_and_loss, has_aux=True)
        self.per_sample_grad_fn = vmap(grad_fn, in_dims=(0, 0, 0, 0))

    def _get_init_weights(self, batch_size: int, device: torch.device):
        """初始化记忆权重（复制模型参数到 batch 维度）。"""
        weights = {}
        for name, p in zip(self.mem_param_names,
                           self.memory_model.parameters()):
            weights[name] = p.unsqueeze(0).expand(batch_size, *p.shape).clone()
        return weights

    def store_memories(
        self,
        seq: torch.Tensor,
        weights: dict,
        momentum_state: dict = None,
    ):
        """
        将 key-value 信息写入记忆模型权重。

        参考: neural_memory.py store_memories() 第568-819行

        流程:
        1. 投影得到 keys, values
        2. 分 chunk（每 chunk 共享一个学习率）
        3. 对每个 chunk 计算记忆模型的 per-sample 梯度
        4. 梯度 × adaptive_lr → surprise（参数更新量）
        5. 通过动量 + 权重衰减 → 最终更新

        Returns:
            weights: 更新后的权重
            momentum_state: 更新后的动量
            surprises: 每个 chunk 的 surprise 值（用于监控）
        """
        batch, seq_len = seq.shape[:2]
        chunk_size = min(self.chunk_size, seq_len)

        # 截断到 chunk_size 的整倍
        usable_len = (seq_len // chunk_size) * chunk_size
        if usable_len == 0:
            return weights, momentum_state, torch.tensor(0.0, device=seq.device)

        seq_usable = seq[:, :usable_len]

        # 归一化
        seq_normed = self.store_norm(seq_usable)

        # Keys / Values
        keys = self.to_keys(seq_normed)
        values = self.to_values(seq_normed)

        # 自适应学习率: sigmoid(linear(x)) * max_lr
        # 参考: neural_memory.py 第623-624行
        adaptive_lr = self.to_adaptive_step(seq_normed).sigmoid()  # [B, L, 1]
        adaptive_lr = adaptive_lr.squeeze(-1)  # [B, L]

        # 权重衰减
        decay_factor = self.to_decay_factor(
            seq_normed.view(batch, -1, chunk_size, self.dim).mean(dim=2)
        ).sigmoid()  # [B, num_chunks, 1]

        # 动量
        momentum_coeff = self.to_momentum(
            seq_normed.view(batch, -1, chunk_size, self.dim).mean(dim=2)
        ).sigmoid()  # [B, num_chunks, 1]

        # 分 chunk
        num_chunks = usable_len // chunk_size
        keys = keys.view(batch, num_chunks, chunk_size, self.dim)
        values = values.view(batch, num_chunks, chunk_size, self.dim)
        adaptive_lr = adaptive_lr.view(batch, num_chunks, chunk_size)

        # 初始化动量
        if momentum_state is None:
            momentum_state = {name: torch.zeros_like(w) for name, w in weights.items()}

        all_surprises = []

        for chunk_idx in range(num_chunks):
            chunk_keys = keys[:, chunk_idx]      # [B, C, D]
            chunk_values = values[:, chunk_idx]   # [B, C, D]
            chunk_lr = adaptive_lr[:, chunk_idx]  # [B, C]

            # 计算 per-sample 梯度
            # 参考: neural_memory.py 第704行
            grads, unweighted_loss = self.per_sample_grad_fn(
                weights, chunk_keys, chunk_lr, chunk_values
            )

            all_surprises.append(unweighted_loss.mean().detach())

            # 更新每个参数
            d = decay_factor[:, chunk_idx]  # [B, 1]
            m = momentum_coeff[:, chunk_idx]  # [B, 1]

            for name in self.mem_param_names:
                g = -grads[name]  # 取负：梯度下降

                # 动量更新
                # 参考: neural_memory.py 第770-779行
                momentum_state[name] = m.view(
                    batch, *([1] * (g.dim() - 1))
                ) * momentum_state[name] + g

                update = momentum_state[name]

                # 权重衰减
                # 参考: neural_memory.py 第801-803行
                # w_new = (1-decay) * w + update
                weights[name] = (
                    1 - d.view(batch, *([1] * (weights[name].dim() - 1)))
                ) * weights[name] + update

        surprise_mean = torch.stack(all_surprises).mean() if all_surprises else torch.tensor(0.0, device=seq.device)
        return weights, momentum_state, surprise_mean

    def retrieve_memories(
        self,
        seq: torch.Tensor,
        weights: dict,
    ) -> torch.Tensor:
        """
        从记忆模型检索信息（vmap 向量化版）。

        参考: neural_memory.py retrieve_memories() 第821-910行

        v2 优化：消除 Python for 循环，使用 vmap 对 functional_call
        做 batch 并行调用。预期速度提升 2-3x。
        """
        batch, seq_len = seq.shape[:2]

        # 归一化
        seq_normed = self.retrieve_norm(seq)

        # 投影为 queries
        queries = self.to_queries(seq_normed)  # [B, L, D]

        # vmap 向量化：一次调用处理所有 batch
        # 单样本检索函数（params 和 query 都没有 batch 维度）
        def _single_retrieve(params, query):
            # query: [L, D] → [1, L, D] 以匹配 nn.Module 的 batch 约定
            return functional_call(
                self.memory_model, params, query.unsqueeze(0)
            ).squeeze(0)  # [L, D]

        # vmap 沿 batch 维度(dim=0)对 params 和 query 并行
        batched_retrieve = vmap(_single_retrieve, in_dims=(0, 0))

        # weights dict 的每个值已经是 [B, ...] 形状，直接传入
        batched_weights = {name: weights[name] for name in self.mem_param_names}
        values = batched_retrieve(batched_weights, queries)  # [B, L, D]

        # 输出归一化 + 门控
        values = self.output_norm(values)
        gate = self.output_gate(seq)  # [B, L, 1]
        values = values * gate

        return values

    def forward(
        self,
        seq: torch.Tensor,
        state: dict = None,
    ) -> tuple:
        """
        完整的 store → retrieve 流程。

        参考: neural_memory.py forward() 第912-1078行

        EGGROLL 兼容：
          store_memories 内部使用 vmap(grad(...))，依赖 autograd 计算图。
          EGGROLL 训练器在 eval 模式下对参数做扰动评估 fitness，
          此时 autograd 图被破坏。因此 eval 模式下跳过 store，
          只用现有权重做 retrieve。

        Args:
            seq: [B, L, D]
            state: (weights, momentum) 或 None

        Returns:
            retrieved: [B, L, D] 检索到的记忆
            next_state: (weights, momentum) 更新后的状态
            surprise: 平均 surprise 值
        """
        batch = seq.shape[0]
        device = seq.device

        # 初始化状态
        if state is None:
            weights = self._get_init_weights(batch, device)
            momentum = None
        else:
            weights, momentum = state

        surprise = torch.tensor(0.0, device=device)

        # 存储（仅训练模式，eval 模式跳过以兼容 EGGROLL 的参数扰动）
        if self.training:
            weights, momentum, surprise = self.store_memories(
                seq, weights, momentum
            )

        # 检索
        retrieved = self.retrieve_memories(seq, weights)

        next_state = (weights, momentum)
        return retrieved, next_state, surprise
