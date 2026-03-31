"""
NEXUS 差分注意力模块

严格参考 Microsoft 官方实现:
  https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py

融合:
1. Microsoft Differential Attention — 官方实现的 lambda 计算方式
2. DeepSeek-V3 MLA — 官方 model.py 中的 KV 低秩压缩路径
3. Apple KV Cache Sharing — 偶数层复用奇数层 KV

关键实现细节（来自官方代码，非论文推测）：
- lambda = exp(q1·k1_sum) - exp(q2·k2_sum) + lambda_init
  其中 lambda_init = 0.8 - 0.6 * exp(-0.3 * depth)
- head_dim = embed_dim // num_heads // 2  (每个head拆成两个sub-head)
- Q 投影 → [2*num_heads, head_dim]
- K 投影 → [2*num_heads, head_dim] (或 kv_heads 用于 GQA)
- V 投影 → [num_heads, 2*head_dim]
- 输出后过 subln (对 2*head_dim 做 RMSNorm)
- 最终乘以 (1 - lambda_init) 缩放
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NexusConfig
from .int8_kv_cache import TurboQuantCompressor


class RMSNorm(nn.Module):
    """RMSNorm — 比 LayerNorm 快（省去均值计算）。与 DeepSeek-V3 源码一致。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def lambda_init_fn(depth: int) -> float:
    """
    官方的 lambda 初始化函数。
    来源: multihead_diffattn.py 第30-31行
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DifferentialAttention(nn.Module):
    """
    差分注意力 + MLA压缩 + KV共享。

    严格遵循官方实现的数学:
    1. Q 拆成 2*num_heads 个 head_dim 维的sub-head
    2. K 同上
    3. V 保持 num_heads 个 2*head_dim 维
    4. 计算两组注意力图并相减
    5. lambda = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + lambda_init
    6. 结果过 subln 后乘以 (1 - lambda_init)
    """

    def __init__(self, config: NexusConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = config.d_model
        self.num_heads = config.n_heads

        # 官方: head_dim = embed_dim // num_heads // 2
        self.head_dim = config.d_model // config.n_heads // 2
        self.scaling = self.head_dim ** -0.5

        # === Q/K/V 投影 ===
        # 判断是否使用 KV 共享（Apple: 偶数层复用奇数层的 KV）
        self.is_kv_shared = config.kv_share_enabled and (layer_idx % 2 == 1)

        # Q: 输出维度 = embed_dim (= 2*num_heads * head_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        if not self.is_kv_shared:
            # === MLA 压缩路径 (DeepSeek-V3 官方做法) ===
            # 步骤1: 联合降维 d_model → latent_dim
            self.kv_down_proj = nn.Linear(
                self.embed_dim, config.mla_latent_dim, bias=False
            )
            # MLA 中间做 RMSNorm（DeepSeek-V3 model.py 第436行）
            self.kv_norm = RMSNorm(config.mla_latent_dim)
            # 步骤2: 上投影回 K 和 V 的联合维度
            # K: [2*num_heads, head_dim] → total = embed_dim
            # V: [num_heads, 2*head_dim] → total = embed_dim
            self.kv_up_proj = nn.Linear(
                config.mla_latent_dim,
                self.embed_dim + self.embed_dim,  # K + V
                bias=False,
            )

        # 输出投影
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # === 官方 lambda 参数 (multihead_diffattn.py 第66-69行) ===
        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )

        # SubLN: 对每个 head 的 2*head_dim 输出做 RMSNorm
        # (multihead_diffattn.py 第71行)
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

        # === 缓存因果掩码（避免每次 forward 重建）===
        self._cached_mask_size = 0

        # === TurboQuant KV Cache 量化（QAT + STE）===
        # 集成到注意力计算的热路径：K/V 计算后、注意力乘法前插入 compress→decompress
        # K 的每个 sub-head dim = head_dim，V 的每个 head dim = 2*head_dim
        # 两者维度不同，需要独立的 compressor
        self.kv_quant_enabled = config.kv_quant_enabled
        if self.kv_quant_enabled and not self.is_kv_shared:
            # KV-shared 层复用上一层的 KV，不需要重复量化
            self.k_compressor = TurboQuantCompressor(
                head_dim=self.head_dim,
                bits=config.kv_quant_key_bits,
                seed=42 + layer_idx,  # 每层不同的旋转矩阵
            )
            self.v_compressor = TurboQuantCompressor(
                head_dim=2 * self.head_dim,
                bits=config.kv_quant_value_bits,
                seed=42 + layer_idx + 1000,
            )

    def _quantize_ste(self, x: torch.Tensor, compressor: TurboQuantCompressor) -> torch.Tensor:
        """
        STE（Straight-Through Estimator）风格的量化。

        量化感知训练（QAT）的标准做法：
          前向：x_out = compress → decompress（引入量化噪声）
          反向：梯度直通，∂L/∂x_in = ∂L/∂x_out（忽略量化的不可微性）

        数学等价于：x_out = x + (quantize(x) - x).detach()
        这里 .detach() 使得量化误差 (quantize(x) - x) 不参与梯度计算，
        但其数值贡献在前向传播中保留。

        Args:
            x: [B, heads, L, dim] 待量化的 KV tensor
            compressor: TurboQuantCompressor 实例

        Returns:
            量化后的 tensor（与输入同 shape，梯度直通）
        """
        compressed = compressor.compress(x)
        x_quantized = compressor.decompress(compressed)
        # STE: 前向用量化值，反向梯度直通原始值
        return x + (x_quantized - x).detach()

    def _compute_kv_via_mla(self, x: torch.Tensor):
        """
        MLA 路径: x → compress → norm → decompress → (K, V)
        参考 DeepSeek-V3 model.py MLA.forward 第474-481行
        """
        bsz, seq_len, _ = x.shape
        # 联合压缩
        compressed = self.kv_down_proj(x)  # [B, L, latent_dim]
        compressed = self.kv_norm(compressed)
        # 联合解压
        kv = self.kv_up_proj(compressed)  # [B, L, 2*embed_dim]
        k, v = kv.split([self.embed_dim, self.embed_dim], dim=-1)

        # K → [B, L, 2*num_heads, head_dim]
        k = k.view(bsz, seq_len, 2 * self.num_heads, self.head_dim)
        # V → [B, L, num_heads, 2*head_dim]
        v = v.view(bsz, seq_len, self.num_heads, 2 * self.head_dim)

        return k, v

    def forward(
        self,
        x: torch.Tensor,
        shared_kv: tuple = None,
    ) -> tuple:
        """
        前向传播。严格遵循 multihead_diffattn.py 第73-126行。

        Args:
            x: [B, L, d_model]
            shared_kv: (K, V) 从上一层共享（Apple KV Sharing）

        Returns:
            output: [B, L, d_model]
            kv_for_sharing: (K, V) 供下一层复用
        """
        bsz, tgt_len, _ = x.size()

        # Q 投影并拆成 2*num_heads 个 sub-head
        q = self.q_proj(x)
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)

        # KV: 使用共享或自行计算
        if self.is_kv_shared and shared_kv is not None:
            k, v = shared_kv
        else:
            k, v = self._compute_kv_via_mla(x)

        # KV Sharing 优化：只在非 shared 层且下一层需要时才 clone
        # 审计优化：偶数层（非 shared）的 KV 才需要给下一层用
        if not self.is_kv_shared:
            kv_for_sharing = (k.clone(), v.clone())
        else:
            kv_for_sharing = shared_kv  # shared 层不产生新 KV

        # 转置为 [B, heads, L, dim]
        q = q.transpose(1, 2)  # [B, 2*nh, L, hd]
        k = k.transpose(1, 2)  # [B, 2*nh, L, hd]
        v = v.transpose(1, 2)  # [B, nh, L, 2*hd]

        # === TurboQuant KV 量化（QAT + STE 直通估计器）===
        # 前向：用量化后的 K/V 计算注意力（模拟推理时的量化误差）
        # 反向：梯度直通原始值（STE，保证训练稳定性）
        if self.kv_quant_enabled and hasattr(self, 'k_compressor'):
            k = self._quantize_ste(k, self.k_compressor)
            v = self._quantize_ste(v, self.v_compressor)

        # 缩放 Q
        q = q * self.scaling

        # 注意力分数
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # [B, 2*nh, L, L]

        # 因果掩码（缓存优化：只在 seq_len 变化时重建）
        if self._cached_mask_size != tgt_len:
            mask = torch.triu(
                torch.zeros(tgt_len, tgt_len, device=x.device).fill_(float("-inf")),
                diagonal=1,
            )
            self.register_buffer('_causal_mask', mask, persistent=False)
            self._cached_mask_size = tgt_len

        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = attn_weights + self._causal_mask

        # Softmax
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).type_as(attn_weights)

        # === 差分核心 (官方第113-117行) ===
        # lambda = exp(sum(λ_q1 * λ_k1)) - exp(sum(λ_q2 * λ_k2)) + lambda_init
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 拆成两组注意力图: [B, nh, 2, L, L]
        attn_weights = attn_weights.view(
            bsz, self.num_heads, 2, tgt_len, tgt_len
        )
        # 差分: attn[:, :, 0] - lambda * attn[:, :, 1]
        diff_attn = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        # 与 V 做加权和: [B, nh, L, 2*hd]
        attn_output = torch.matmul(diff_attn, v)

        # SubLN (官方第121行)
        attn_output = self.subln(attn_output)

        # 缩放 (官方第122行)
        attn_output = attn_output * (1 - self.lambda_init)

        # 合并多头: [B, L, nh * 2 * hd] = [B, L, embed_dim]
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * 2 * self.head_dim
        )

        # 输出投影
        attn_output = self.out_proj(attn_output)

        return attn_output, kv_for_sharing
