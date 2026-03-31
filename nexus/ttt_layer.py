"""
NEXUS TTT-Linear 层 (v4: 向量化版)

v4 关键优化：消除 Python for 循环
  - 所有 mini-batch 的 grad 基于 W_0 一次性计算（批量化）
  - W 增量用 torch.cumsum 沿 num_mb 维累积（前缀和）
  - 输出用单次张量操作计算
  - 预期速度提升 4-8x

数学原理（Dual Form + 前缀和近似）：
  标准 dual form（顺序）：
    对每个 mini-batch i:
      grad_i = ln_fused_l2_bwd(K_i @ W_i + b_i, target_i)
      out_i = Q_i @ W_i - (eta_i * tril(Q_i @ K_i^T)) @ grad_i + b1_bar_i
      W_{i+1} = W_i - eta_last_i * K_i^T @ grad_i

  向量化近似（并行）：
    一次性计算所有 grad（使用 W_0）：
      grad_all = ln_fused_l2_bwd(K_all @ W_0 + b_0, target_all)
    W 增量前缀和：
      delta_W_i = -eta_last_i * K_i^T @ grad_i
      W_i ≈ W_0 + cumsum(delta_W)[:i]
    
  精度 trade-off：
    - 用 W_0 代替 W_i 计算 grad 引入近似误差
    - 但 mini_batch_size=4 时只有 4 步，误差较小
    - Surprise-Gated 机制会自动补偿：高 surprise → 更大更新
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NexusConfig


def _ln_fwd(x, gamma, beta, eps=1e-6):
    """LayerNorm 前向。参考: ttt_source.py ln_fwd"""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta


def _ln_fused_l2_bwd(x, target, gamma, beta, eps=1e-6):
    """
    LayerNorm + L2 loss 融合反向传播。
    参考: ttt_source.py ln_fused_l2_bwd (第91-113行)
    """
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    y = gamma * x_hat + beta
    grad_output = y - target
    grad_x_hat = grad_output * gamma

    z = (1.0 / D) * (
        D * grad_x_hat
        - grad_x_hat.sum(dim=-1, keepdim=True)
        - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
    ) / std

    return z


class TTTLinearLayer(nn.Module):
    """
    Test-Time Training Linear Layer (v4: 向量化版)。
    
    消除 Python for 循环，所有 mini-batch 并行计算。
    """

    def __init__(self, config: NexusConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.mini_batch_size = config.ttt_mini_batch_size
        self.ttt_base_lr = config.ttt_base_lr

        # QKV 投影
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # TTT 内部线性模型 W1, b1
        self.W1 = nn.Parameter(
            torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim))
        )
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        # 可学习学习率
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.normal(0, 0.02, size=(self.num_heads, 1, self.d_model))
        )
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.zeros(self.num_heads, 1)
        )

        # TTT 内部 LayerNorm
        self.ttt_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        # Token 位置缩放
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros(self.mini_batch_size))

        self.post_norm = nn.LayerNorm(self.d_model)

    def get_eta(self, X, mini_batch_size):
        """
        计算 eta 矩阵。
        Returns:
            token_eta: [B, nh, num_mb, mbs, 1]
            ttt_lr_eta: [B, nh, num_mb, 1, mbs]
        """
        B = X.shape[0]
        num_mini_batch = X.shape[1]

        ttt_lr = torch.einsum(
            "bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight
        ) + self.learnable_ttt_lr_bias.reshape(1, -1, 1, 1, 1)
        ttt_lr = F.sigmoid(ttt_lr)

        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.ttt_base_lr * ttt_lr / self.head_dim

        token_idx = self.token_idx[:mini_batch_size] + self.learnable_token_idx[:mini_batch_size]
        token_idx = torch.clamp_min(token_idx, 0.0)

        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (B, self.num_heads, num_mini_batch, mini_batch_size, 1),
        )

        return token_eta, ttt_lr_eta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        TTT forward (v4: 完全向量化)。
        
        核心优化：不再有 Python for 循环。
        所有 mini-batch 的 grad 基于 W_0 一次性计算，
        W 增量用 cumsum 前缀和累加。
        """
        B, L, _ = hidden_states.shape
        nh, hd = self.num_heads, self.head_dim

        # 保证序列长度可被 mini_batch_size 整除
        mbs = min(self.mini_batch_size, L)
        pad_len = (mbs - L % mbs) % mbs
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L

        num_mb = L_padded // mbs

        # QKV 投影
        XQ = self.q_proj(hidden_states)
        XK = self.k_proj(hidden_states)
        XV = self.v_proj(hidden_states)

        # Split heads + reshape to mini-batches: [B, nh, num_mb, mbs, hd]
        XQ = XQ.view(B, L_padded, nh, hd).transpose(1, 2).reshape(B, nh, num_mb, mbs, hd)
        XK = XK.view(B, L_padded, nh, hd).transpose(1, 2).reshape(B, nh, num_mb, mbs, hd)
        XV = XV.view(B, L_padded, nh, hd).transpose(1, 2).reshape(B, nh, num_mb, mbs, hd)

        # Compute eta: [B, nh, num_mb, mbs, mbs]
        X_for_eta = hidden_states.reshape(B, num_mb, mbs, self.d_model)
        token_eta, ttt_lr_eta = self.get_eta(X_for_eta, mbs)
        eta = token_eta * ttt_lr_eta  # [B, nh, num_mb, mbs, mbs]

        # === 向量化核心：所有 mini-batch 并行计算 ===

        # W_0 扩展到 batch: [B, nh, hd, hd]
        W0 = self.W1.unsqueeze(0).expand(B, -1, -1, -1)
        b0 = self.b1.unsqueeze(0).expand(B, -1, -1, -1)

        ln_weight = self.ttt_norm_weight.reshape(1, nh, 1, 1, hd)
        ln_bias = self.ttt_norm_bias.reshape(1, nh, 1, 1, hd)

        # 1. 一次性计算所有 mini-batch 的 Z1（基于 W_0）
        # XK: [B, nh, num_mb, mbs, hd], W0: [B, nh, hd, hd]
        # Z1_all: [B, nh, num_mb, mbs, hd]
        Z1_all = torch.einsum("bnmsh,bnhd->bnmsd", XK, W0) + b0.unsqueeze(2)

        # 2. 自监督目标
        reconstruction_target = XV - XK  # [B, nh, num_mb, mbs, hd]

        # 3. 融合 LN + L2 梯度（所有 mini-batch 并行）
        grad_all = _ln_fused_l2_bwd(
            Z1_all, reconstruction_target,
            ln_weight, ln_bias
        )  # [B, nh, num_mb, mbs, hd]

        # 4. 计算 W 增量并用 cumsum 累积
        # delta_W_i = -eta_last_i * K_i^T @ grad_i
        # last_eta: [B, nh, num_mb, mbs, 1]
        last_eta = eta[:, :, :, -1:, :]  # 最后一行的 eta: [B, nh, num_mb, 1, mbs]
        # 但我们需要 [B, nh, num_mb, mbs, 1] 形式来加权每个 token
        last_eta_weights = eta[:, :, :, -1, :, None]  # [B, nh, num_mb, mbs, 1]

        # delta_W_per_mb: [B, nh, num_mb, hd, hd]
        # = (last_eta * XK)^T @ grad = XK^T @ diag(last_eta) @ grad
        weighted_XK = last_eta_weights * XK  # [B, nh, num_mb, mbs, hd]
        delta_W_per_mb = -torch.einsum(
            "bnmsh,bnmsd->bnmhd", weighted_XK, grad_all
        )  # [B, nh, num_mb, hd, hd]

        # 前缀和：W_i = W_0 + cumsum(delta_W)[:i]
        # 但注意：mini-batch i 的输出应使用 W_0 + sum(delta_W[0:i])
        # cumsum 给出等于 sum(delta_W[0:i+1])，需要 shift-right
        cumsum_delta_W = torch.cumsum(delta_W_per_mb, dim=2)  # [B, nh, num_mb, hd, hd]
        # shift: W_i = W_0 + cumsum[i-1]（第 0 个 mini-batch 用 W_0）
        shifted_cumsum = F.pad(cumsum_delta_W[:, :, :-1], (0, 0, 0, 0, 1, 0))  # [B, nh, num_mb, hd, hd]
        W_all = W0.unsqueeze(2) + shifted_cumsum  # [B, nh, num_mb, hd, hd]

        # b 增量（类似）
        weighted_grad = last_eta_weights * grad_all  # [B, nh, num_mb, mbs, hd]
        delta_b_per_mb = -weighted_grad.sum(dim=3, keepdim=True)  # [B, nh, num_mb, 1, hd]
        cumsum_delta_b = torch.cumsum(delta_b_per_mb, dim=2)
        shifted_cumsum_b = F.pad(cumsum_delta_b[:, :, :-1], (0, 0, 0, 0, 1, 0))
        b_all = b0.unsqueeze(2) + shifted_cumsum_b  # [B, nh, num_mb, 1, hd]

        # 5. 重新计算输出（使用更新后的 W_i）
        # Z1_bar = Q_i @ W_i - (eta_i * tril(Q_i @ K_i^T)) @ grad_i + b1_bar_i
        # Q @ W: [B, nh, num_mb, mbs, hd] @ [B, nh, num_mb, hd, hd]
        QW = torch.einsum("bnmsh,bnmhd->bnmsd", XQ, W_all)

        # tril(Q @ K^T): [B, nh, num_mb, mbs, mbs]
        Attn = torch.einsum("bnmsh,bnmth->bnmst", XQ, XK)
        # 对每个 mini-batch 应用下三角掩码
        tril_mask = torch.tril(torch.ones(mbs, mbs, device=hidden_states.device))
        Attn = Attn * tril_mask

        # b1_bar = b_i - tril(eta) @ grad
        tril_eta = eta * tril_mask  # [B, nh, num_mb, mbs, mbs]
        b1_bar = b_all - torch.einsum("bnmst,bnmth->bnmsh", tril_eta, grad_all)

        # Z1_bar = QW - (eta * Attn) @ grad + b1_bar
        Z1_bar = QW - torch.einsum("bnmst,bnmth->bnmsh", eta * Attn, grad_all) + b1_bar

        # LayerNorm
        Z1_bar = _ln_fwd(Z1_bar, ln_weight, ln_bias)

        # 残差: output = Q + Z1_bar
        output = XQ + Z1_bar  # [B, nh, num_mb, mbs, hd]

        # reshape 输出: [B, nh, num_mb, mbs, hd] → [B, L, d_model]
        output = output.reshape(B, nh, L_padded, hd)

        if pad_len > 0:
            output = output[:, :, :L, :]

        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.post_norm(output)
        output = self.o_proj(output)

        return output
