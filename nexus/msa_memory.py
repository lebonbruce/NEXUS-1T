"""
MSA: Memory Sparse Attention — 适配持续学习的 1 亿 KV 记忆引擎

参考:
  - EverMind MSA 论文 (arxiv:2603.23516)
  - GitHub: EverMind-AI/MSA

核心原理:
  1. 压缩记忆库: 文档 KV → 分块均值池化 → INT8 量化 → 紧凑存储
  2. 分层存储: 路由键 K̄ᴿ 驻 GPU，内容 K̄/V̄ 驻 CPU
  3. 稀疏路由: Q → 路由投影 → 与 K̄ᴿ 余弦相似度 → Top-k 选择
  4. 上下文组装: 检索到的 K̄/V̄ 与本地 KV 拼接 → 稀疏注意力
  5. 文档级 RoPE: 每个文档位置从 0 开始

持续学习适配:
  MSA 原版: 离线编码语料库 → 推理时路由
  我们的适配: 每个任务训练结束 → 压缩存入 MemoryBank → 新任务检索旧知识
  "文档" = "已学任务的 KV 快照"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .int8_kv_cache import INT8Quantizer


class ChunkCompressor(nn.Module):
    """
    分块均值池化压缩器。

    参考 MSA 论文 Section 2.1:
      K̄ = mean_pool(K, chunk_size=P)
      V̄ = mean_pool(V, chunk_size=P)
      K̄ᴿ = mean_pool(Kᴿ, chunk_size=P)

    压缩比: L/P（seq_len / chunk_size）
    """

    def __init__(self, chunk_size: int = 4):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        分块均值池化。

        Args:
            x: [..., L, D] 任意前缀维度的 tensor

        Returns:
            [..., L//P, D] 压缩后的 tensor
        """
        # 获取序列长度（倒数第二维）
        seq_len = x.shape[-2]
        dim = x.shape[-1]
        prefix_shape = x.shape[:-2]

        # 对齐到 chunk_size 的倍数（截断尾部不完整块）
        n_chunks = seq_len // self.chunk_size
        if n_chunks == 0:
            # 序列太短，直接取均值压缩为 1 个块
            return x.mean(dim=-2, keepdim=True)

        usable_len = n_chunks * self.chunk_size
        x_trimmed = x[..., :usable_len, :]

        # reshape → 分块 → 均值
        x_chunked = x_trimmed.reshape(*prefix_shape, n_chunks, self.chunk_size, dim)
        return x_chunked.mean(dim=-2)  # [..., n_chunks, D]


class RoutingProjector(nn.Module):
    """
    路由键投影器。

    参考 MSA 论文 Section 2.3:
      独立于主注意力的 QK 投影，专门用于记忆路由。
      Q_router 和 K_router 都是学习的投影层。

    为什么需要独立投影:
      主注意力的 QK 是为 token-level 细粒度匹配优化的，
      路由需要的是 document-level 粗粒度语义匹配。
      独立投影允许路由学习不同粒度的相似度空间。
    """

    def __init__(self, d_model: int, d_route: int):
        """
        Args:
            d_model: 模型维度
            d_route: 路由键维度（可以比 d_model 小以节省内存）
        """
        super().__init__()
        self.q_router = nn.Linear(d_model, d_route, bias=False)
        self.k_router = nn.Linear(d_model, d_route, bias=False)

    def project_query(self, x: torch.Tensor) -> torch.Tensor:
        """投影查询用于路由。x: [B, L, D] → [B, L, d_route]"""
        return self.q_router(x)

    def project_key(self, x: torch.Tensor) -> torch.Tensor:
        """投影键用于路由存储。x: [B, L, D] → [B, L, d_route]"""
        return self.k_router(x)


class MemoryBank:
    """
    压缩记忆库 — MSA 的核心数据结构。

    存储结构:
      - routing_keys: 压缩路由键 K̄ᴿ (GPU, float16/float32)
      - content_keys: 压缩内容键 K̄ (CPU, INT8)
      - content_values: 压缩内容值 V̄ (CPU, INT8)
      - doc_ids: 文档/任务 ID

    分层存储策略 (MSA 论文 Section 2.2):
      路由键常驻 GPU（用于快速匹配）
      内容 KV 存 CPU（仅 Top-k 选中时异步拉取到 GPU）
      → 总显存与语料库大小解耦
    """

    def __init__(self, max_docs: int = 100, device: torch.device = None):
        self.max_docs = max_docs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantizer = INT8Quantizer()

        # GPU 常驻：路由键（float，用于快速余弦相似度）
        self.routing_keys: list[torch.Tensor] = []    # [n_chunks, d_route]

        # CPU 存储：内容 KV（INT8，仅被选中时拉取到 GPU）
        self.content_k_quantized: list[torch.Tensor] = []
        self.content_k_scales: list[torch.Tensor] = []
        self.content_v_quantized: list[torch.Tensor] = []
        self.content_v_scales: list[torch.Tensor] = []

        # 元数据
        self.doc_ids: list[int] = []

    def ingest(
        self,
        routing_key: torch.Tensor,
        content_key: torch.Tensor,
        content_value: torch.Tensor,
        doc_id: int,
    ):
        """
        将一个文档/任务的 KV 压缩存入记忆库。

        Args:
            routing_key: [n_chunks, d_route] 压缩后的路由键 (float)
            content_key: [n_chunks, D] 压缩后的内容键 (float)
            content_value: [n_chunks, D] 压缩后的内容值 (float)
            doc_id: 文档/任务标识符

        所有输入已经过 ChunkCompressor 压缩。
        """
        # 路由键常驻 GPU（小尺寸，用于快速匹配）
        self.routing_keys.append(routing_key.detach().to(self.device))

        # 内容 KV 量化后存 CPU
        k_q, k_s = self.quantizer.quantize(content_key.detach())
        v_q, v_s = self.quantizer.quantize(content_value.detach())
        self.content_k_quantized.append(k_q.cpu())
        self.content_k_scales.append(k_s.cpu())
        self.content_v_quantized.append(v_q.cpu())
        self.content_v_scales.append(v_s.cpu())
        self.doc_ids.append(doc_id)

        # FIFO 淘汰
        while len(self.routing_keys) > self.max_docs:
            self.routing_keys.pop(0)
            self.content_k_quantized.pop(0)
            self.content_k_scales.pop(0)
            self.content_v_quantized.pop(0)
            self.content_v_scales.pop(0)
            self.doc_ids.pop(0)

    def route(
        self,
        query_routing: torch.Tensor,
        top_k: int = 4,
    ) -> tuple[list[int], torch.Tensor]:
        """
        稀疏路由：从记忆库中选择最相关的 Top-k 文档。

        参考 MSA 论文 Section 2.3:
          1. Q_route 与每个文档的 K̄ᴿ 做余弦相似度
          2. 对注意力头取均值（我们简化为单头）
          3. 对查询 token 取最大值 → 每文档得分
          4. 全局 Top-k 选择

        Args:
            query_routing: [B, L, d_route] 查询路由向量
            top_k: 选择的文档数

        Returns:
            selected_indices: Top-k 文档在记忆库中的索引
            scores: 对应的相似度分数
        """
        if not self.routing_keys:
            return [], torch.tensor([])

        # query 聚合：对 batch 和 token 做均值，得到一个 query 向量
        # [B, L, d_route] → [d_route]
        q = query_routing.mean(dim=(0, 1))  # 全局查询表示
        q = F.normalize(q, dim=-1)

        # 与每个文档的路由键做余弦相似度
        scores = []
        for rk in self.routing_keys:
            # rk: [n_chunks, d_route]
            rk_norm = F.normalize(rk, dim=-1)
            # 每个 chunk 与 query 的相似度
            chunk_scores = torch.matmul(rk_norm, q)  # [n_chunks]
            # 取最大值作为文档得分（MSA: max pooling over chunks）
            doc_score = chunk_scores.max().item()
            scores.append(doc_score)

        scores_tensor = torch.tensor(scores, device=self.device)

        # Top-k 选择
        actual_k = min(top_k, len(scores))
        top_scores, top_indices = torch.topk(scores_tensor, actual_k)

        return top_indices.tolist(), top_scores

    def fetch_content(
        self,
        indices: list[int],
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从 CPU 拉取选中文档的 K̄/V̄ 到 GPU。

        Args:
            indices: 要拉取的文档索引
            device: 目标设备

        Returns:
            keys: [total_chunks, D] 反量化的内容键
            values: [total_chunks, D] 反量化的内容值
        """
        if not indices:
            return None, None

        target_device = device or self.device

        keys_list = []
        values_list = []

        for idx in indices:
            k = self.quantizer.dequantize(
                self.content_k_quantized[idx],
                self.content_k_scales[idx],
            ).to(target_device)
            v = self.quantizer.dequantize(
                self.content_v_quantized[idx],
                self.content_v_scales[idx],
            ).to(target_device)
            keys_list.append(k)
            values_list.append(v)

        keys = torch.cat(keys_list, dim=-2)
        values = torch.cat(values_list, dim=-2)

        return keys, values

    @property
    def num_docs(self) -> int:
        return len(self.routing_keys)

    def memory_stats(self) -> dict:
        """返回内存使用统计。"""
        routing_bytes = sum(rk.nelement() * 4 for rk in self.routing_keys)
        content_bytes = sum(
            kq.nelement() + ks.nelement() * 4
            for kq, ks in zip(self.content_k_quantized, self.content_k_scales)
        ) + sum(
            vq.nelement() + vs.nelement() * 4
            for vq, vs in zip(self.content_v_quantized, self.content_v_scales)
        )
        return {
            'num_docs': self.num_docs,
            'routing_bytes_gpu': routing_bytes,
            'content_bytes_cpu': content_bytes,
            'total_bytes': routing_bytes + content_bytes,
        }


class MSALayer(nn.Module):
    """
    Memory Sparse Attention 层 — 集成到 NEXUS Block 的记忆注入点。

    参考 MSA 论文:
      仅在模型后半层激活（早期层做本地处理，后期层做全局检索）。
      检索到的 K̄/V̄ 与本地 KV 拼接，用于稀疏注意力。

    持续学习适配:
      - MemoryBank 存储已学任务的 KV 快照（INT8）
      - 路由器学习将新输入与旧任务匹配
      - 检索到的旧知识通过注意力注入当前前向传播
    """

    def __init__(self, d_model: int, d_route: int = None,
                 chunk_size: int = 4, top_k: int = 4,
                 max_docs: int = 100):
        super().__init__()
        d_route = d_route or d_model
        self.d_model = d_model
        self.top_k = top_k

        # 压缩器
        self.compressor = ChunkCompressor(chunk_size)

        # 路由投影器
        self.router = RoutingProjector(d_model, d_route)

        # 记忆库
        self.memory_bank = MemoryBank(max_docs=max_docs)

        # 检索到的记忆与当前表示的融合层
        # 使用门控机制控制记忆注入强度
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # 记忆注意力的 QKV 投影（独立于主注意力）
        self.mem_q_proj = nn.Linear(d_model, d_model, bias=False)

        # 审计修复：独立的 K/V 投影层，解决 K=V bug
        # 原实现中 compressed_k 和 compressed_v 都是直接对 h 做 mean-pool，完全相同
        # 修复：用不同投影生成 K 和 V，实现真正的路由+内容分离
        self.content_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.content_v_proj = nn.Linear(d_model, d_model, bias=False)

    def ingest_task_memory(
        self,
        hidden_states: torch.Tensor,
        task_id: int,
    ):
        """
        将一个任务的隐层表示压缩存入记忆库。

        在 on_task_end() 时调用：
        1. 对 hidden_states 做 chunk compression
        2. 生成路由键
        3. INT8 量化存入 MemoryBank

        Args:
            hidden_states: [B, L, D] 任务训练结束时的最终隐层表示
            task_id: 任务 ID
        """
        with torch.no_grad():
            # 取 batch 均值作为任务的典型表示
            h = hidden_states.mean(dim=0, keepdim=True)  # [1, L, D]

            # 审计修复：用独立投影生成 K 和 V（原实现 K=V 导致注意力退化）
            compressed_k = self.compressor(self.content_k_proj(h)).squeeze(0)  # [n_chunks, D]
            compressed_v = self.compressor(self.content_v_proj(h)).squeeze(0)  # [n_chunks, D]

            # 路由键
            routing_key = self.router.project_key(h)
            routing_key = self.compressor(routing_key).squeeze(0)  # [n_chunks, d_route]

            # 存入记忆库
            self.memory_bank.ingest(
                routing_key=routing_key,
                content_key=compressed_k,
                content_value=compressed_v,
                doc_id=task_id,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MSA 前向传播：路由 → 检索 → 注入。

        Args:
            x: [B, L, D] 当前层的隐层表示

        Returns:
            [B, L, D] 注入记忆后的表示
        """
        if self.memory_bank.num_docs == 0:
            # 没有记忆可检索，直接返回零向量
            return torch.zeros_like(x)

        B, L, D = x.shape
        device = x.device

        # 1. 路由：生成查询路由向量
        q_route = self.router.project_query(x)  # [B, L, d_route]

        # 2. 选择 Top-k 文档
        selected_indices, scores = self.memory_bank.route(
            q_route, top_k=self.top_k
        )

        if not selected_indices:
            return torch.zeros_like(x)

        # 3. 从 CPU 拉取内容到 GPU
        mem_k, mem_v = self.memory_bank.fetch_content(
            selected_indices, device=device
        )

        # mem_k, mem_v: [n_chunks_total, D]
        # 扩展 batch 维度
        mem_k = mem_k.unsqueeze(0).expand(B, -1, -1)  # [B, n_chunks, D]
        mem_v = mem_v.unsqueeze(0).expand(B, -1, -1)

        # 4. 稀疏注意力：当前表示 attend to 检索到的记忆
        q = self.mem_q_proj(x)  # [B, L, D]

        # 简化注意力（不用多头，因为这是辅助记忆注入）
        scale = D ** -0.5
        attn_scores = torch.bmm(q * scale, mem_k.transpose(-2, -1))  # [B, L, n_chunks]
        attn_weights = F.softmax(attn_scores, dim=-1)
        mem_output = torch.bmm(attn_weights, mem_v)  # [B, L, D]

        # 5. 门控融合
        gate_input = torch.cat([x, mem_output], dim=-1)  # [B, L, 2D]
        gate = self.memory_gate(gate_input)  # [B, L, D] (sigmoid 0-1)
        output = gate * mem_output

        return output
