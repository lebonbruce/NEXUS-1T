"""
NEXUS vs Baseline GPT — 模型定义（v4: Scale-Aware，组件按规模自动启/停）

v4 改动（基于 3000 步预训练 + 消融实验的数据驱动决策）：
  - MLA KV 压缩：根据 d_model 自动调整压缩比
      d_model < 512:   不压缩（独立 KV 投影）← 消融证实 96 维瓶颈有害
      d_model 512-1023: 2x 压缩
      d_model >= 1024:  4x 压缩（DeepSeek-V3 原始设计点）
  - TTT-Linear：根据 d_model 和 seq_len 自动决定是否启用
      d_model < 512 或 seq_len < 1024: 跳过（消融证实 ROI 不足）
      d_model >= 512 且 seq_len >= 1024: 启用
  - DiffAttn：始终启用（计算量 = 标准 Attn，论文多规模有效）
  - SwiGLU：始终启用（Llama/Mistral 标配）

消融实验证据摘要：
  MLA 4x 压缩在 d_model=384 时: +0.4172 loss (+8.8%) — 最大拖累
  SwiGLU 在 1000 步时: +0.1165 (+2.4%) — 收敛偏慢但长训练可恢复
  TTT 在 seq=512 时: -0.0919 (-1.8%) — 微弱改善，但速度代价 7.74x

两个模型公平对比：
  BaselineGPT: RoPE + FlashAttn MHA + GELU FFN
  NexusGPT:    RoPE + FlashAttn DiffAttn(+可选MLA) + 可选TTT + SwiGLU FFN
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Scale-Aware 配置决策函数
# ============================================================

def get_mla_compression(d_model: int) -> int:
    """
    根据 d_model 决定 MLA KV 压缩比。

    决策依据（消融实验 + 论文分析）：
      - d_model=384 时，4x 压缩（96维瓶颈）导致 +8.8% loss 
        → K 和 V 共享 96 维空间无法编码足够的匹配性 + 内容信息
      - DeepSeek-V3 (d_model=7168) 用 4x 压缩后仍有 1792 维，信息充足
      - 经验法则：KV 瓶颈 >= 256 维时压缩才不伤害质量

    Returns:
        compression_ratio: 1 表示不压缩，2 表示 2x，4 表示 4x
    """
    kv_latent_at_4x = d_model // 4
    kv_latent_at_2x = d_model // 2

    if kv_latent_at_4x >= 256:
        # d_model >= 1024: 4x 压缩安全（瓶颈 >= 256 维）
        return 4
    elif kv_latent_at_2x >= 192:
        # d_model >= 384: 2x 压缩（瓶颈 >= 192 维）
        # 注意：d_model=384 时 2x 给 192 维，比 4x 的 96 维好很多
        return 2
    else:
        # d_model < 384: 不压缩
        return 1


def should_enable_ttt(d_model: int, seq_len: int) -> bool:
    """
    根据 d_model 和 seq_len 决定是否启用 TTT-Linear。

    决策依据（消融实验 + 论文分析）：
      - TTT 的核心价值是"在线适应"：需要足够长的上下文才能累积有效梯度
      - seq_len=512 时仅 32 个 mini-batch，TTT 效果微弱（-1.8%）但速度代价 7.74x
      - Stanford 论文在 seq_len=4096+ 时验证 TTT 优势
      - TTT 的 per-sample 权重矩阵 [B, n_batches, D, D] 在 D 较大时内存更合理

    Returns:
        True 如果应该启用 TTT
    """
    # TTT 需要足够的 mini-batch 数量来累积有意义的梯度
    # 以 mini_batch_size=16 计算，至少需要 64 个 mini-batch
    min_batches_for_ttt = 64
    effective_batches = seq_len // 16

    # 同时需要模型有足够的容量来利用 TTT 的适应能力
    return d_model >= 512 and effective_batches >= min_batches_for_ttt


# ============================================================
# 共享组件
# ============================================================

class RMSNorm(nn.Module):
    """RMSNorm — Llama/DeepSeek 标配。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================
# RoPE（旋转位置编码 — Llama/Qwen/DeepSeek 全系标配）
# ============================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoFormer, 2021; Llama 标准实现)。

    优势 vs learned position embedding:
      1. 无最大长度限制（可外推到训练长度之外）
      2. 相对位置信息（而非绝对位置）
      3. 参数量为零（纯数学计算）

    Scale-up 关键：从 512 → 2048 → 8192 无需重新训练。
    """
    def __init__(self, dim, max_seq_len=8192, base=10000.0):
        super().__init__()
        # θ_i = 1 / (base^(2i/dim)), i = 0, 1, ..., dim/2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        # 复制一份以匹配 head_dim（前半和后半用相同频率）
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x):
    """RoPE 辅助：将 x 的前后半互换并取负。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    将 RoPE 应用到 Q 和 K。

    q, k: [B, n_heads, T, head_dim]
    cos, sin: [T, head_dim] → broadcast 到 [1, 1, T, head_dim]
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, hd]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


# ============================================================
# Baseline: 标准 MHA（RoPE + Flash Attention）
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    标准因果自注意力（Baseline 用）。
    使用 RoPE + F.scaled_dot_product_attention (Flash Attention 2)。
    """
    def __init__(self, d_model, n_heads, seq_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=seq_len * 2)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        cos, sin = self.rope(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flash Attention 2（PyTorch 2.0+，自动选择最优内核）
        drop_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


# ============================================================
# NEXUS: DiffAttn + 可选 MLA（Scale-Aware，RoPE + Flash Attention）
# ============================================================

class DiffAttnMLA(nn.Module):
    """
    差分注意力 + 规模自适应 KV 投影（Scale-Aware 版）。

    DiffAttn (Microsoft, NeurIPS 2024):
      A_diff = softmax(Q1·K1^T)V - λ·softmax(Q2·K2^T)V
      使用 Flash Attention 分别计算两个子注意力，再差分合并。

    MLA (DeepSeek-V3, 2024) — 仅在大模型启用:
      KV 共享低秩瓶颈：x → kv_latent(压缩) → K, V(展开)
      小模型（d_model<512）不压缩，避免信息瓶颈。

    Scale-Aware KV 投影策略：
      compression=1: K 和 V 各自独立投影（等价于标准 Attention 的 K/V 路径）
      compression=2: 2x 低秩压缩（瓶颈 = d_model//2）
      compression=4: 4x 低秩压缩（DeepSeek-V3 原始设计）
    """
    def __init__(self, d_model, n_heads, seq_len, layer_idx=0, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        # 差分注意力的 head_dim 是标准的一半
        self.head_dim = d_model // n_heads // 2
        self.d_model = d_model
        self.dropout = dropout

        # Q: 标准投影（所有规模都一样）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # Scale-Aware KV 投影
        self.mla_compression = get_mla_compression(d_model)

        if self.mla_compression > 1:
            # MLA 模式：KV 共享低秩瓶颈
            self.d_kv_latent = d_model // self.mla_compression
            self.kv_down_proj = nn.Linear(d_model, self.d_kv_latent, bias=False)
            self.k_up_proj = nn.Linear(self.d_kv_latent, d_model, bias=False)
            self.v_up_proj = nn.Linear(self.d_kv_latent, d_model, bias=False)
        else:
            # 独立 KV 投影模式：K 和 V 各自有完整的 d_model → d_model 投影
            # 信息容量 = 2 * d_model（最大化）
            self.d_kv_latent = d_model  # 标记用，不实际参与计算
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # DiffAttn λ 参数
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx + 1))

        # SubLN
        self.subln = RMSNorm(2 * self.head_dim)

        # RoPE（基于 sub-head dim）
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=seq_len * 2)

    def forward(self, x):
        B, T, _ = x.size()
        nh = self.n_heads
        hd = self.head_dim

        # Q: [B, T, d_model] → [B, 2*nh, T, hd]
        q = self.q_proj(x).view(B, T, 2 * nh, hd).transpose(1, 2)

        # KV 投影：根据压缩策略分支
        if self.mla_compression > 1:
            # MLA 低秩压缩路径
            kv_latent = self.kv_down_proj(x)
            k = self.k_up_proj(kv_latent).view(B, T, 2 * nh, hd).transpose(1, 2)
            v = self.v_up_proj(kv_latent).view(B, T, nh, 2 * hd).transpose(1, 2)
        else:
            # 独立 KV 投影路径（最大信息容量）
            k = self.k_proj(x).view(B, T, 2 * nh, hd).transpose(1, 2)
            v = self.v_proj(x).view(B, T, nh, 2 * hd).transpose(1, 2)

        # RoPE 应用到 Q 和 K 的每个 sub-head
        cos, sin = self.rope(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 拆分 Q, K 为两组 sub-heads
        q1, q2 = q[:, :nh], q[:, nh:]  # 各 [B, nh, T, hd]
        k1, k2 = k[:, :nh], k[:, nh:]

        # Flash Attention: 分别计算两个子注意力
        drop_p = self.dropout if self.training else 0.0
        out1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True, dropout_p=drop_p)
        out2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True, dropout_p=drop_p)

        # 计算可学习的 λ
        lambda_val = (torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
                      - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
                      + self.lambda_init)

        # 差分注意力: output1 - λ·output2
        output = out1 - lambda_val * out2  # [B, nh, T, 2hd]
        output = self.subln(output)
        output = output * (1 - self.lambda_init)

        output = output.transpose(1, 2).contiguous().view(B, T, nh * 2 * hd)
        return self.out_proj(output)


# ============================================================
# GELU FFN（Baseline 用）
# ============================================================

class GELUMLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ============================================================
# SwiGLU FFN（NEXUS 用）
# ============================================================

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoESwiGLUFFN(nn.Module):
    """
    Mixture-of-Experts SwiGLU FFN（DeepSeek-V3 风格）。

    每个 token 只激活 top_k 个 expert，大幅减少计算量。
    总参数 = n_experts * single_FFN_params，但每个 token 的计算量
    仅为 top_k/n_experts 的比例。

    包含 load balancing 辅助损失，防止 expert 崩塔（所有 token 都去同一个 expert）。
    """
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2, dropout=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model

        # Router: 小型线性层，将 token 投影到 n_experts 维的路由分数
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # N 个独立的 SwiGLU expert
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, d_ff, dropout) for _ in range(n_experts)
        ])

        # Load balancing 系数
        self.lb_alpha = 0.01

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]
        N = x_flat.size(0)

        # Router 计算路由分数
        router_logits = self.router(x_flat)  # [N, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)  # [N, n_experts]

        # Top-k 选择
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)  # [N, top_k]
        # 归一化 top-k 权重（让选中的 expert 权重和为 1）
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # [N, top_k]

        # 初始化输出
        output = torch.zeros_like(x_flat)

        # 逐 expert 处理（比逐 token 更高效，因为同一 expert 的 token 可以 batch）
        for i, expert in enumerate(self.experts):
            # 找到选择了这个 expert 的 token
            # topk_indices: [N, top_k], 我们要找所有 (token_idx, slot) 其中 topk_indices[token_idx, slot] == i
            mask = (topk_indices == i)  # [N, top_k]
            if not mask.any():
                continue

            # 获取选中该 expert 的 token 索引和权重
            token_indices = mask.any(dim=-1).nonzero(as_tuple=True)[0]  # 哪些 token 选了这个 expert
            # 每个 token 在该 expert 上的权重（取 mask 为 True 的那个 slot 的权重）
            weights_for_expert = (topk_weights * mask.float()).sum(dim=-1)[token_indices]  # [n_selected]

            expert_output = expert(x_flat[token_indices])  # [n_selected, D]
            output[token_indices] += weights_for_expert.unsqueeze(-1) * expert_output

        # Load balancing 辅助损失（存储在 self 上，让训练循环可以访问）
        # f_i = fraction of tokens routed to expert i
        # P_i = average routing probability for expert i
        # loss = alpha * n_experts * sum(f_i * P_i)
        if self.training:
            f = torch.zeros(self.n_experts, device=x.device)
            for i in range(self.n_experts):
                f[i] = (topk_indices == i).float().sum() / (N * self.top_k)
            P = router_probs.mean(dim=0)  # [n_experts]
            self.aux_loss = self.lb_alpha * self.n_experts * (f * P).sum()
        else:
            self.aux_loss = 0.0

        return output.view(B, T, D)


# ============================================================
# TTT-Linear（cumsum 向量化，完整版）
# ============================================================

class TTTLinear(nn.Module):
    """
    Test-Time Training Linear v5 — Scale-Up 安全版。

    核心设计（Stanford TTT 论文）：
      h_t = (W_0 + Σ_{s<t} Δ_s) · x_t
      严格因果：cum_grad_shifted = [0, cum_grad[0], ..., cum_grad[N-2]]

    v5 新增（Scale-Up 三修复）：
      1. FP32 cumsum: 防止 BF16 精度雪崩（测试发现 max_err=1124x）
      2. LoRA 低秩模式: ttt_rank 非 None 时，W 拆为 A[D,r] + B[r,D]
         显存从 O(B*N*D²) 降为 O(B*N*D*r)，d=2048 不再 OOM
      3. 梯度低秩投影: LoRA 模式下梯度也走低秩路径，彻底避免 D×D
    """
    def __init__(self, d_model, mini_batch_size=16, ttt_lr=5e-4, ttt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.mini_batch_size = mini_batch_size
        self.ttt_base_lr = ttt_lr
        self.ttt_rank = ttt_rank

        self.theta_proj = nn.Linear(d_model, d_model, bias=False)

        if ttt_rank is None:
            # 全秩模式：W ∈ R^{D×D}（原始 TTT）
            self.W = nn.Linear(d_model, d_model, bias=False)
        else:
            # LoRA 低秩模式：W ≈ B @ A
            # A ∈ R^{r×D}（下投影）, B ∈ R^{D×r}（上投影）
            # 显存: 2*D*r 代替 D*D（当 r=D//4 时节省 87.5%）
            self.W_A = nn.Linear(d_model, ttt_rank, bias=False)   # x → low-rank
            self.W_B = nn.Linear(ttt_rank, d_model, bias=False)   # low-rank → output
            # 初始化：A 使用 kaiming，B 初始为零（初始 W ≈ 0，对称性好）
            nn.init.zeros_(self.W_B.weight)

        # 学习率门控（per mini-batch 自适应 lr）
        self.lr_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.output_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm = RMSNorm(d_model)

    def _get_W0(self):
        """获取初始权重矩阵 W_0 [D, D]。"""
        if self.ttt_rank is None:
            return self.W.weight
        else:
            # W_0 = W_B.weight @ W_A.weight : [D, r] @ [r, D] = [D, D]
            return self.W_B.weight @ self.W_A.weight

    def _apply_W0(self, x):
        """高效应用 W_0（LoRA 模式不用实例化 D×D 矩阵）。"""
        if self.ttt_rank is None:
            return self.W(x)
        else:
            return self.W_B(self.W_A(x))

    def forward(self, x):
        B, T, D = x.shape
        mbs = self.mini_batch_size
        n_batches = T // mbs

        if n_batches == 0:
            return self.norm(self._apply_W0(x) * self.output_gate(x))

        x_trimmed = x[:, :n_batches * mbs]
        remainder = x[:, n_batches * mbs:]
        x_mb = x_trimmed.view(B, n_batches, mbs, D)

        target = self.theta_proj(x_mb)
        W_0 = self._get_W0()  # [D, D]

        # 一次 matmul 计算所有 mini-batch 的预测
        pred = torch.matmul(x_mb, W_0.t())
        error = pred - target
        grad_all = torch.matmul(error.transpose(-1, -2), x_mb) / mbs
        grad_all = torch.clamp(grad_all, -1.0, 1.0)

        # 学习率调制
        x_mean = x_mb.mean(dim=2)
        lr_mod = self.lr_gate(x_mean)
        effective_lr = (self.ttt_base_lr
                        * lr_mod.mean(-1, keepdim=True).unsqueeze(-1))

        # === FP32 安全 cumsum（修复 BF16 精度雪崩）===
        scaled_grad = effective_lr * grad_all
        input_dtype = scaled_grad.dtype
        cum_grad_raw = torch.cumsum(scaled_grad.float(), dim=1).to(input_dtype)

        # 因果移位：位置 0 = 零，位置 i = cum_grad_raw[:, i-1]
        cum_grad_causal = torch.zeros_like(cum_grad_raw)
        cum_grad_causal[:, 1:] = cum_grad_raw[:, :-1]

        W_all = W_0.unsqueeze(0).unsqueeze(0) - cum_grad_causal

        # batched matmul
        output = torch.matmul(x_mb, W_all.transpose(-1, -2))
        output = output * self.output_gate(x_mb)
        result = output.reshape(B, n_batches * mbs, D)

        # 余数处理
        if remainder.size(1) > 0:
            W_final = W_0.unsqueeze(0) - cum_grad_raw[:, -1]
            rem_out = torch.bmm(remainder, W_final.transpose(-1, -2))
            rem_out = rem_out * self.output_gate(remainder)
            result = torch.cat([result, rem_out], dim=1)

        return self.norm(result)


# ============================================================
# Baseline GPT（RoPE + Flash MHA + GELU FFN）
# ============================================================

class BaselineBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = GELUMLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineGPT(nn.Module):
    """标准 GPT 架构（RoPE + Flash Attention）。无 learned pos_emb。"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # RoPE 替代 learned position embedding（零参数位置编码）
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            BaselineBlock(d_model, n_heads, d_ff, seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        # 只有 token embedding（RoPE 在 attention 内部应用）
        h = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# NEXUS GPT（Scale-Aware: DiffAttn + 可选MLA + 可选TTT + SwiGLU）
# ============================================================

class NexusBlock(nn.Module):
    """
    NEXUS Block — Scale-Aware 版。

    根据 d_model 和 seq_len 自动决定组件配置：
      - DiffAttn: 始终启用
      - MLA: d_model >= 384 时启用低秩压缩（比例随 d_model 自动调整）
      - TTT: d_model >= 512 且 seq_len >= 1024 时启用
      - SwiGLU: 始终启用

    启用 TTT 时: LN → DiffAttn → + → LN → TTT → + → LN → SwiGLU → +
    禁用 TTT 时: LN → DiffAttn → + → LN → SwiGLU → +
    """
    def __init__(self, d_model, n_heads, d_ff, seq_len, layer_idx, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = DiffAttnMLA(d_model, n_heads, seq_len, layer_idx, dropout)

        # Scale-Aware TTT 启用决策
        self.use_ttt = should_enable_ttt(d_model, seq_len)

        if self.use_ttt:
            self.ln2 = RMSNorm(d_model)
            # LoRA-TTT: 始终使用低秩模式（数据驱动决策）
            # rank=8 在所有实验中表现最优（比 full rank 好 8x）
            # 低秩 = 正则化：限制 W 只在最重要的语义流形上更新
            ttt_rank = min(8, d_model // 4)
            self.ttt = TTTLinear(d_model, mini_batch_size=16, ttt_rank=ttt_rank)
            self.ln3 = RMSNorm(d_model)
        else:
            self.ln2 = RMSNorm(d_model)
            # 不创建 TTT 层 → 不浪费参数和计算

        # Scale-Aware FFN + MoE
        # 50M: dense SwiGLU | 70B+: MoE 32-64 experts | 1T: MoE 256 experts
        # 当前 prototype 仅在 d>=1024 启用 8 experts（足够验证机制）
        self.use_moe = d_model >= 1024
        if self.use_moe:
            self.ffn = MoESwiGLUFFN(d_model, d_ff, n_experts=8, top_k=2, dropout=dropout)
        else:
            self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.use_ttt:
            x = x + self.ttt(self.ln2(x))
            x = x + self.ffn(self.ln3(x))
        else:
            x = x + self.ffn(self.ln2(x))
        return x


class NexusGPT(nn.Module):
    """
    NEXUS GPT — Scale-Aware 预训练架构（v4）。

    核心原则："不要在小模型上强加大模型的优化技术"
    
    各组件根据模型规模自动启停：
      始终启用: DiffAttn + SwiGLU（任何规模都有效）
      条件启用:
        MLA KV 压缩: d_model < 512 → 不压缩; 512-1023 → 2x; >= 1024 → 4x
        TTT-Linear:  d_model >= 512 且 seq_len >= 1024 → 启用

    构建时会打印配置摘要，方便确认组件状态。
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            NexusBlock(d_model, n_heads, d_ff, seq_len, i, dropout)
            for i in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

        # 打印 Scale-Aware 配置摘要
        mla_ratio = get_mla_compression(d_model)
        ttt_on = should_enable_ttt(d_model, seq_len)
        mla_latent = d_model // mla_ratio if mla_ratio > 1 else d_model
        print(f"  [NexusGPT Scale-Aware Config]")
        print(f"    d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
        print(f"    DiffAttn:  ON  (always)")
        print(f"    MLA:       {'OFF (独立KV)' if mla_ratio == 1 else f'ON ({mla_ratio}x 压缩, 瓶颈={mla_latent}维)'}")
        print(f"    TTT:       {'ON' if ttt_on else 'OFF'} (d_model{'≥' if d_model>=512 else '<'}512, seq_len{'≥' if seq_len>=1024 else '<'}1024)")
        moe_on = d_model >= 1024
        print(f"    FFN:       {'MoE-SwiGLU (8 experts, top-2)' if moe_on else 'SwiGLU (dense)'}")
        print(f"    Params:    {self.count_params():,}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        h = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 兼容性 Checkpoint 加载器
# ============================================================

def load_compatible_checkpoint(model, path, device='cpu'):
    """
    智能兼容性加载器：只加载名称匹配且形状一致的权重，跳过不兼容部分。

    使用场景：
      1. v3 → v4 迁移（MLA 4x→2x, TTT 移除）
      2. Baseline → NEXUS 迁移（复用 tok_emb, head 等共享层）
      3. 不同 d_model/d_ff 的模型之间迁移

    原理：
      PyTorch 的 strict=False 只能处理 missing/extra keys，
      但 shape mismatch（同名不同维度）仍会报错。
      本函数手动逐 key 过滤，只加载 name + shape 都一致的参数。

    Args:
        model: 目标模型实例
        path: checkpoint 文件路径
        device: 加载到的设备（建议先加载到 cpu 再 .to(cuda)）

    Returns:
        model: 部分加载后的模型
        stats: dict，包含 loaded/mismatched/ignored 统计
    """
    if not os.path.exists(path):
        print(f"  ❌ Checkpoint 不存在: {path}")
        return model, {"loaded": 0, "mismatched": 0, "ignored": 0}

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    # 兼容被包装在 dict 中的 checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    compatible_state = {}
    stats = {"loaded": 0, "mismatched": 0, "ignored": 0}
    mismatched_keys = []

    for key, param in state_dict.items():
        if key in model_state:
            if param.shape == model_state[key].shape:
                compatible_state[key] = param
                stats["loaded"] += 1
            else:
                stats["mismatched"] += 1
                mismatched_keys.append(
                    f"      {key}: ckpt={list(param.shape)} vs model={list(model_state[key].shape)}"
                )
        else:
            stats["ignored"] += 1

    # 加载过滤后的权重（strict=False 因为会有 missing keys）
    model.load_state_dict(compatible_state, strict=False)

    # 报告
    total = stats["loaded"] + stats["mismatched"] + stats["ignored"]
    print(f"  📦 [Checkpoint Loader] {path}")
    print(f"     ✅ 成功加载: {stats['loaded']}/{total} 个参数块")
    if stats["mismatched"] > 0:
        print(f"     ⚠️  形状冲突 (随机初始化): {stats['mismatched']} 个")
        for line in mismatched_keys[:5]:  # 最多显示 5 个
            print(line)
        if len(mismatched_keys) > 5:
            print(f"      ... 及其他 {len(mismatched_keys)-5} 个")
    if stats["ignored"] > 0:
        print(f"     🗑️  旧层已移除 (忽略): {stats['ignored']} 个")

    return model, stats
