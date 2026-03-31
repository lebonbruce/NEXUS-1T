"""
NEXUS: Neural EXtensible Unified System — 主模型

组装所有组件的完整 Transformer 模型:
  1. Differential Attention (Microsoft) — 噪声消除的注意力
  2. MLA KV 压缩 (DeepSeek-V3) — 低秩KV缓存
  3. Apple KV Cache Sharing — 偶数层复用奇数层KV
  4. TTT-Linear (Stanford) — 推理时自监督学习
  5. Dynamic MoE FFN (SCE + DeepSeek-V3) — 共享专家 + 动态生长
  6. FFT Token Mixing (FNet) — O(N log N) 无参数预混合
  7. Neural Memory (Google Titans) — surprise驱动的持久化记忆

继承 CLModel 接口以兼容现有 SCE benchmark 框架。

参考实现:
  - SCE: sce/models/sce_model.py (CLModel 接口, SurpriseMonitor 生长)
  - DeepSeek-V3: model.py Block 类 (norm → attn → norm → ffn)
  - Titans: neural_memory.py NeuralMemory (store → retrieve)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sce.models.base import CLModel
from sce.models.components.surprise_growth import SurpriseMonitor

from .config import NexusConfig
from .diff_attention import DifferentialAttention, RMSNorm
from .ttt_layer import TTTLinearLayer
from .moe_ffn import DynamicMoEFFN
from .neural_memory import NeuralMemory
from .msa_memory import MSALayer


class FFTMixing(nn.Module):
    """
    FNet 风格的 FFT Token Mixing。

    参考: Google FNet 论文 + HuggingFace transformers FNetModel

    核心实现（来自多个 GitHub 实现的共识）：
      output = Real(FFT_seq(FFT_hidden(x)))

    O(N log N) 的无参数 token 混合层，
    作为注意力前的预处理，帮助跨 token 信息扩散。
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿 hidden 维度做 FFT
        x_fft = torch.fft.fft(x, dim=-1)
        # 沿 sequence 维度做 FFT
        x_fft = torch.fft.fft(x_fft, dim=-2)
        # 取实部
        return x_fft.real


class NEXUSBlock(nn.Module):
    """
    NEXUS Transformer 块。

    数据流:
      x → FFT Mixing (可选) → RMSNorm → DiffAttn → residual
        → RMSNorm → TTT-Linear → residual
        → RMSNorm → Dynamic MoE FFN → residual

    参考 DeepSeek-V3 Block 类 (model.py 第704-743行):
      x = x + attn(attn_norm(x))
      x = x + ffn(ffn_norm(x))
    """

    def __init__(self, config: NexusConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # === FFT 预混合 (FNet) ===
        self.fft_mixing = FFTMixing() if config.fft_mixing_enabled else None
        self.fft_norm = RMSNorm(config.d_model) if config.fft_mixing_enabled else None

        # === Differential Attention + MLA + KV Sharing ===
        self.attn_norm = RMSNorm(config.d_model)
        self.diff_attn = DifferentialAttention(config, layer_idx)

        # === TTT-Linear (推理时自监督学习) ===
        self.ttt_norm = RMSNorm(config.d_model)
        self.ttt_layer = TTTLinearLayer(config)

        # === Dynamic MoE FFN (共享专家 + 生长专家) ===
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = DynamicMoEFFN(config)

        # === MSA Memory Injection (后半层激活) ===
        # 参考 MSA 论文：早期层做本地处理，后期层做全局检索
        self.msa_layer = None
        if layer_idx >= config.msa_inject_after_layer:
            self.msa_layer = MSALayer(
                d_model=config.d_model,
                d_route=config.msa_route_dim,
                chunk_size=config.msa_compression_chunk,
                top_k=config.msa_top_k,
                max_docs=config.msa_memory_capacity,
            )
            self.msa_norm = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        task_id: int = None,
        shared_kv: tuple = None,
    ) -> tuple:
        """
        Args:
            x: [B, L, d_model]
            task_id: 当前任务ID（用于专家路由）
            shared_kv: 上一层的 (K, V)（Apple KV Sharing）

        Returns:
            x: [B, L, d_model]
            kv_for_sharing: (K, V) 供下一层复用
        """
        # [1] FFT 预混合（无参数，O(N log N)）
        if self.fft_mixing is not None:
            x = x + self.fft_mixing(self.fft_norm(x))

        # [2] Differential Attention（短期精确检索）
        attn_out, kv_for_sharing = self.diff_attn(self.attn_norm(x), shared_kv)
        x = x + attn_out

        # [3] TTT-Linear（中期自监督学习，条件激活）
        # 审计优化：短序列时 TTT 只有 4 个 mini-batch，信噫比极低
        seq_len = x.shape[1]
        if seq_len > self.config.ttt_min_seq_len:
            x = x + self.ttt_layer(self.ttt_norm(x))

        # [4] Dynamic MoE FFN（容量扩展）
        x = x + self.ffn(self.ffn_norm(x), task_id=task_id)

        # [5] MSA Memory Injection（后半层，记忆检索注入）
        if self.msa_layer is not None:
            x = x + self.msa_layer(self.msa_norm(x))

        return x, kv_for_sharing


class NEXUSTransformer(CLModel):
    """
    NEXUS 完整模型。

    架构总览:
      Token Embedding + Position Embedding + Task Embedding
      → [NEXUSBlock × n_layers] (每2层注入一次 Neural Memory)
      → RMSNorm → LM Head

    继承 CLModel 以兼容 SCE benchmark 的统一训练/评估接口。

    关键持续学习机制:
    1. TTT: 推理时自监督学习（W 跨 mini-batch 更新）
    2. SurpriseFFN: 物理切分隔离任务知识
    3. Neural Memory: surprise 驱动的长期记忆持久化
    4. KD (Knowledge Distillation): 新专家继承旧专家知识
    """

    def __init__(self, config: NexusConfig):
        super().__init__(config)
        self.config = config

        # === Embedding 层 ===
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)

        # === NEXUS Blocks ===
        self.blocks = nn.ModuleList([
            NEXUSBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])

        # === Neural Memory (每 N 层注入一次) ===
        # 参考: titans 的 Memory-as-Context 架构
        num_memories = max(1, config.n_layers // config.titans_inject_every_n)
        self.neural_memories = nn.ModuleList([
            NeuralMemory(config) for _ in range(num_memories)
        ])
        # 记录哪些层后面注入 memory
        self.memory_inject_layers = [
            (i + 1) * config.titans_inject_every_n - 1
            for i in range(num_memories)
        ]

        # === 输出层 ===
        self.final_norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

        # === Surprise Monitor (SCE 框架) ===
        self.surprise_monitor = SurpriseMonitor(
            alpha=config.surprise_ema_alpha,
            sigma=config.surprise_sigma,
            warmup=config.surprise_warmup,
        )
        self._kd_alpha = config.kd_alpha
        self._has_old_expert = False

        # Neural Memory 状态 (跨 forward 持久化)
        self._mem_states = [None] * num_memories

        # === Replay Buffer (经验回放) ===
        # 用户独立测试证明 Replay 是最有效的抗遗忘手段 (74.8% AA, -0.1% BWT)
        self._buffer_x: list[torch.Tensor] = []
        self._buffer_y: list[torch.Tensor] = []
        self._buffer_tasks: list[int] = []

    @staticmethod
    def _detach_state(state):
        """Detach memory state 以防止计算图跨步骤泄露。"""
        if state is None:
            return None
        weights, momentum = state
        detached_w = {k: v.detach() for k, v in weights.items()}
        detached_m = None
        if momentum is not None:
            detached_m = {k: v.detach() for k, v in momentum.items()}
        return (detached_w, detached_m)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        B, T = x.size()
        device = x.device

        # Embedding
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)

        # 逐层前向
        shared_kv = None
        mem_idx = 0

        for layer_idx, block in enumerate(self.blocks):
            h, shared_kv = block(h, task_id=task_id, shared_kv=shared_kv)

            # Neural Memory 注入（条件激活：短序列跳过）
            # 审计优化：vmap(grad(...)) 是 44x 慢的头号杀手（占 60-70%）
            if layer_idx in self.memory_inject_layers and mem_idx < len(self.neural_memories):
                if T > self.config.titans_min_seq_len:
                    # 训练时使用持久化状态，评估时用 None（避免 batch_size 不匹配）
                    current_state = self._mem_states[mem_idx] if self.training else None
                    mem_output, new_state, _ = self.neural_memories[mem_idx](
                        h, state=current_state
                    )
                    if self.training:
                        # detach 防止计算图跨步骤泄漏
                        self._mem_states[mem_idx] = self._detach_state(new_state)
                    h = h + mem_output
                mem_idx += 1

        # 输出
        return self.head(self.final_norm(h))

    def check_and_grow(self, loss_value: float):
        """Surprise 检测和自动生长。"""
        should_grow = self.surprise_monitor.update(loss_value)
        if should_grow:
            self._do_grow()

    def _do_grow(self):
        """所有 FFN 层同时分裂新细胞。"""
        # 审计优化：深拷贝初始化 → 相似度=1.0 → 必然合并，当前阶段可选关闭
        if not self.config.moe_growth_enabled:
            return
        device = next(self.parameters()).device
        for block in self.blocks:
            block.ffn.grow(device)
        self._has_old_expert = True
        num_experts = len(self.blocks[0].ffn.experts)
        print(f"    [NEXUS] Growing new experts. "
              f"Total experts per layer: {num_experts}")

    def on_task_start(self, task_id: int):
        """新任务开始: 条件生长 + 重置 surprise + 重置 memory 状态。"""
        self.surprise_monitor.reset_trigger()
        if task_id > 0 and self.config.moe_growth_enabled:
            self._do_grow()
        # 注册 task → expert 映射
        for block in self.blocks:
            block.ffn.register_task(task_id)
        # 重置 neural memory 状态（新任务 = 新的记忆环境）
        # 注意: 不完全清除，让 memory 模型权重保持（长期知识）
        # 只清除 momentum（短期优化状态）
        for i in range(len(self._mem_states)):
            if self._mem_states[i] is not None:
                weights, _ = self._mem_states[i]
                self._mem_states[i] = (weights, None)  # 保留权重，清除动量

    def on_task_end(self, task_id: int, train_x: torch.Tensor,
                    train_y: torch.Tensor):
        """任务结束: 注册映射 + 存储 Replay 样本 + MSA 记忆存储。"""
        for block in self.blocks:
            block.ffn.register_task(task_id)

        # 存储 Replay 样本 (参考: replay.py on_task_end)
        n = min(self.config.replay_buffer_per_task, len(train_x))
        idx = torch.randperm(len(train_x))[:n]
        self._buffer_x.append(train_x[idx].cpu())
        self._buffer_y.append(train_y[idx].cpu())
        self._buffer_tasks.append(task_id)

        # === MSA: 将任务知识存入 Memory Bank ===
        # 取一小批训练数据做 forward，获取隐层表示作为任务的 KV 快照
        with torch.no_grad():
            device = next(self.parameters()).device
            sample_n = min(32, len(train_x))
            sample_idx = torch.randperm(len(train_x))[:sample_n]
            sample_x = train_x[sample_idx].to(device)

            # 获取最后一层之前的隐层表示
            B, T = sample_x.size()
            pos = torch.arange(T, device=device)
            task_emb = torch.full((B,), task_id, dtype=torch.long, device=device)
            h = self.token_emb(sample_x) + self.pos_emb(pos) + self.task_emb(task_emb).unsqueeze(1)

            shared_kv = None
            for block in self.blocks:
                h, shared_kv = block(h, task_id=task_id, shared_kv=shared_kv)

            # 将隐层表示存入每个有 MSA 的 block 的记忆库
            for block in self.blocks:
                if block.msa_layer is not None:
                    block.msa_layer.ingest_task_memory(h, task_id)

    def get_replay_data(self, batch_size: int):
        """
        从 buffer 中随机采样一个旧任务的 mini-batch。
        参考: replay.py get_replay_data
        """
        if not self._buffer_x:
            return None
        buf_idx = torch.randint(0, len(self._buffer_x), (1,)).item()
        bx = self._buffer_x[buf_idx]
        by = self._buffer_y[buf_idx]
        task = self._buffer_tasks[buf_idx]
        n = min(batch_size, len(bx))
        sample_idx = torch.randint(0, len(bx), (n,))
        device = next(self.parameters()).device
        return bx[sample_idx].to(device), by[sample_idx].to(device), task

    def forward_with_kd(self, x: torch.Tensor, task_id: int):
        """
        带 KD 的前向传播（兼容 SCE runner）。

        KD 策略: 新任务专家的输出不要偏离旧专家太远。
        参考: sce_model.py 第130-168行
        """
        B, T = x.size()
        device = x.device

        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)

        kd_loss = torch.tensor(0.0, device=device)
        kd_count = 0
        shared_kv = None
        mem_idx = 0

        for layer_idx, block in enumerate(self.blocks):
            # FFT + DiffAttn + TTT
            if block.fft_mixing is not None:
                h = h + block.fft_mixing(block.fft_norm(h))

            attn_out, shared_kv = block.diff_attn(block.attn_norm(h), shared_kv)
            h = h + attn_out

            # TTT 条件跳过（与 forward 一致）
            if T > self.config.ttt_min_seq_len:
                h = h + block.ttt_layer(block.ttt_norm(h))

            # FFN with KD
            h_ffn_in = block.ffn_norm(h)
            ffn_out = block.ffn(h_ffn_in, task_id=task_id)

            if self._has_old_expert and self.training:
                old_out = block.ffn.get_old_expert_output(h_ffn_in)
                if old_out is not None:
                    kd_loss = kd_loss + F.mse_loss(ffn_out, old_out)
                    kd_count += 1

            h = h + ffn_out

            # Neural Memory（条件激活，与 forward 一致）
            if layer_idx in self.memory_inject_layers and mem_idx < len(self.neural_memories):
                if T > self.config.titans_min_seq_len:
                    mem_output, new_state, _ = self.neural_memories[mem_idx](
                        h, state=self._mem_states[mem_idx]
                    )
                    self._mem_states[mem_idx] = self._detach_state(new_state)
                    h = h + mem_output
                mem_idx += 1

        logits = self.head(self.final_norm(h))

        if kd_count > 0:
            kd_loss = self._kd_alpha * kd_loss / kd_count

        return logits, kd_loss
