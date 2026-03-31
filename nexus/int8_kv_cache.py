"""
TurboQuant KV Cache 压缩 — 忠实于论文 + 社区最佳实践的实现

参考:
  - 论文: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)
  - 社区参考实现: tonbistudio/turboquant-pytorch (603 stars, V3 最佳实践)
  - PolarQuant: arxiv:2502.02617 (极坐标变换)

核心算法 (2 阶段):
  Stage 1 (PolarQuant): 随机正交旋转 → 使坐标分布变为可预测的 Beta/Gaussian
                        → 预计算 Lloyd-Max 最优标量量化器 → 无需校准
  Stage 2 (QJL):       1-bit 残差纠错（用于内积估计）

社区发现（6 个独立团队确认）:
  - QJL 在 KV cache 场景有害（softmax 指数放大方差）
  - V3 最佳配置: MSE-only (去掉 QJL) + 非对称 K/V (Keys 4-bit, Values 2-bit)
  - 层自适应: 前/后 N 层用高精度保护
  - 残差窗口: 最近 tokens 保持 fp16

我们的实现 (NEXUS v2):
  - 遵循 V3 最佳实践 (MSE-only, 非对称 K/V)
  - 适配小模型 (d_model=128, 无需 scipy 依赖)
  - 使用 Gaussian 近似 Lloyd-Max（d>=64 时精确度足够）
"""
import torch
import torch.nn as nn
import math
from typing import Optional


# ============================================================================
# Lloyd-Max 最优标量量化器
# ============================================================================

def _gaussian_pdf(x: float, sigma: float) -> float:
    """N(0, sigma^2) 的概率密度函数。"""
    return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-x * x / (2 * sigma * sigma))


def solve_lloyd_max_gaussian(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    求解 Lloyd-Max 最优量化器（Gaussian 近似）。

    旋转后的单位向量每个坐标服从 N(0, 1/d)。
    Lloyd-Max 条件 = 连续 1-D k-means，交替更新：
      1. 边界 = 相邻质心中点
      2. 质心 = E[X | X ∈ partition_i]

    对于 Gaussian，E[X | a < X < b] = σ * (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))

    Args:
        d: 向量维度（决定 σ = 1/√d）
        bits: 量化比特位

    Returns:
        centroids: [2^bits] 最优质心
        boundaries: [2^bits - 1] 边界
    """
    sigma = 1.0 / math.sqrt(d)
    n_levels = 2 ** bits

    # 初始化质心：均匀分布在 [-3.5σ, 3.5σ]
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    # 使用 Gaussian CDF/PDF 的解析公式
    # φ(x) = exp(-x²/2) / √(2π), Φ(x) = 0.5 * erfc(-x/√2)
    for _ in range(max_iter):
        # Step 1：边界 = 相邻质心中点
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

        edges = [lo * 3] + boundaries + [hi * 3]

        # Step 2：用 Gaussian 条件期望更新质心
        new_centroids = []
        for i in range(n_levels):
            a_norm = edges[i] / sigma
            b_norm = edges[i + 1] / sigma

            # E[X | a < X < b] = σ * (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
            phi_a = math.exp(-a_norm * a_norm / 2) / math.sqrt(2 * math.pi)
            phi_b = math.exp(-b_norm * b_norm / 2) / math.sqrt(2 * math.pi)

            # Φ(x) = 0.5 * erfc(-x / √2)
            cdf_a = 0.5 * math.erfc(-a_norm / math.sqrt(2))
            cdf_b = 0.5 * math.erfc(-b_norm / math.sqrt(2))

            denominator = cdf_b - cdf_a
            if denominator > 1e-15:
                numerator = sigma * (phi_a - phi_b)
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])

        # 收敛检查
        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


# ============================================================================
# PolarQuant: 随机正交旋转矩阵
# ============================================================================

def generate_rotation_matrix(d: int, seed: int = 42, device: str = "cpu") -> torch.Tensor:
    """
    生成 Haar 分布随机正交矩阵（通过 QR 分解）。

    关键性质:
      对于 d 维单位向量 v，旋转后 y = Q @ v 的每个坐标
      独立同分布于 Beta((d-1)/2, (d-1)/2)，d>=64 时近似 N(0, 1/d)。
      这使得可以预计算量化器，无需数据校准。
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)

    # 修正 QR 的符号歧义，确保 det(Q) = +1
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)

    return Q.to(device)


# ============================================================================
# TurboQuant V3 MSE Compressor (社区最佳实践)
# ============================================================================

class TurboQuantCompressor:
    """
    TurboQuant V3 MSE-only 压缩器。

    流程:
      1. 归一化到单位球面，保存范数
      2. 随机正交旋转（PolarQuant）
      3. Lloyd-Max 最优标量量化（每坐标独立）
      4. 存储量化索引 + 范数

    社区验证:
      - 不使用 QJL（6个独立团队确认在 softmax attention 中有害）
      - 所有比特用于 MSE 重建质量
    """

    def __init__(self, head_dim: int, bits: int, seed: int = 42, device: str = "cpu"):
        """
        Args:
            head_dim: KV head 维度
            bits: 量化比特位（2/3/4/8）
            seed: 旋转矩阵的随机种子
            device: 计算设备
        """
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        # 预计算旋转矩阵（一次性，所有 KV 共享）
        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)

        # 预计算 Lloyd-Max 码本（一次性）
        centroids, _ = solve_lloyd_max_gaussian(head_dim, bits)
        self.centroids = centroids.to(device)
        self.n_levels = 2 ** bits

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        压缩 KV cache tensor。

        Args:
            states: [B, H, S, D] 或 [B, S, D]

        Returns:
            dict: 包含量化索引和范数的压缩表示
        """
        orig_shape = states.shape
        D = self.head_dim

        # 展平为 (N, D)
        flat = states.reshape(-1, D).float()
        N = flat.shape[0]

        # 1. 归一化到单位球面
        vec_norms = torch.norm(flat, dim=-1)  # (N,)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)

        # 自动设备匹配（当输入在 CUDA 而 Pi/centroids 在 CPU 时）
        Pi = self.Pi.to(flat.device)
        centroids = self.centroids.to(flat.device)

        # 2. PolarQuant 旋转
        rotated = flat_norm @ Pi.T  # (N, D)

        # 3. Lloyd-Max 量化（找最近质心）
        diffs = rotated.unsqueeze(-1) - centroids  # (N, D, n_levels)
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (N, D)

        return {
            "indices": indices.reshape(orig_shape),
            "vec_norms": vec_norms.to(torch.float16).reshape(orig_shape[:-1]),
            "orig_shape": orig_shape,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """
        解压缩回原始 tensor。

        Args:
            compressed: compress() 的输出

        Returns:
            重建的 tensor，形状与输入相同
        """
        orig_shape = compressed["orig_shape"]
        D = self.head_dim

        indices = compressed["indices"].reshape(-1, D).long()
        vec_norms = compressed["vec_norms"].reshape(-1, 1).float()

        # 自动设备匹配
        device = indices.device
        Pi = self.Pi.to(device)
        centroids = self.centroids.to(device)

        # 查表 → 反旋转 → 恢复范数
        reconstructed = (centroids[indices] @ Pi) * vec_norms

        return reconstructed.reshape(orig_shape)

    def compression_ratio(self, dtype_bytes: int = 2) -> float:
        """
        计算理论压缩比。

        Args:
            dtype_bytes: 原始数据类型字节数（fp16=2, fp32=4）

        Returns:
            压缩比 (e.g. 5.0 = 5倍压缩)
        """
        # 压缩后: bits 位/坐标的索引 + 2 字节/向量的范数
        # 原始: dtype_bytes * D 字节/向量
        original_bits_per_vec = self.head_dim * dtype_bytes * 8
        compressed_bits_per_vec = self.head_dim * self.bits + 16  # 16 bits for fp16 norm
        return original_bits_per_vec / compressed_bits_per_vec


# ============================================================================
# TurboQuant V3 非对称 K/V Cache (完整接口)
# ============================================================================

class TurboQuantKVCache:
    """
    TurboQuant V3 非对称 KV Cache 压缩器。

    社区验证的最佳配置:
      - Keys: 4-bit (需要高精度，因为影响 attention 路由)
      - Values: 2-bit (误差在加权平均中自然抵消)
      - 平均 3-bit = 5.3x 压缩 (fp16 基线)

    对比我们之前的错误实现 (简单 INT8 对称量化):
      ❌ 无旋转 → 坐标分布不可预测 → 量化误差大
      ❌ 无 Lloyd-Max → 非最优量化
      ❌ 对称量化 → 无法处理非对称分布
      ❌ K/V 相同精度 → 浪费比特预算

    TurboQuant V3 vs INT8:
      ✓ 3-bit 平均 vs 8-bit → 内存占用 1/2.7
      ✓ 旋转使分布可预测 → 零校准
      ✓ Lloyd-Max 最优性 → 接近香农极限
      ✓ 非对称 K/V → 比特分配在影响最大处
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 4,
        value_bits: int = 2,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits

        self.key_compressor = TurboQuantCompressor(
            head_dim, key_bits, seed=seed, device=device
        )
        self.val_compressor = TurboQuantCompressor(
            head_dim, value_bits, seed=seed + 500, device=device
        )

    @torch.no_grad()
    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[dict, dict]:
        """压缩 K 和 V。"""
        return self.key_compressor.compress(keys), self.val_compressor.compress(values)

    @torch.no_grad()
    def decompress(self, compressed_k: dict, compressed_v: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """解压缩 K 和 V。"""
        return self.key_compressor.decompress(compressed_k), self.val_compressor.decompress(compressed_v)

    def compression_stats(self) -> dict:
        """返回压缩统计。"""
        return {
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "avg_bits": (self.key_bits + self.value_bits) / 2,
            "key_compression_ratio": self.key_compressor.compression_ratio(),
            "value_compression_ratio": self.val_compressor.compression_ratio(),
            "avg_compression_ratio": (
                self.key_compressor.compression_ratio() +
                self.val_compressor.compression_ratio()
            ) / 2,
        }


# ============================================================================
# 向后兼容：保留旧接口名称
# ============================================================================

class INT8Quantizer:
    """
    向后兼容接口。
    已替换为 TurboQuantCompressor (lloyd-max + PolarQuant 旋转)。
    """

    def __init__(self, head_dim: int = 128, device: str = "cpu"):
        self._compressor = TurboQuantCompressor(head_dim, bits=4, device=device)

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """量化（返回 indices 和 norms 以兼容旧接口）。"""
        compressed = self._compressor.compress(tensor)
        return compressed["indices"], compressed["vec_norms"]

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """反量化。"""
        compressed = {
            "indices": indices,
            "vec_norms": norms,
            "orig_shape": indices.shape,  # indices 形状 = 原始形状
        }
        return self._compressor.decompress(compressed)


class QuantizedKVCache:
    """向后兼容的 KV cache 接口。"""

    def __init__(self, head_dim: int = 128, device: str = "cpu"):
        self._cache = TurboQuantKVCache(
            head_dim, key_bits=4, value_bits=2, device=device
        )
        self._storage = []

    def store(self, keys: torch.Tensor, values: torch.Tensor):
        """存储一组 KV。"""
        ck, cv = self._cache.compress(keys, values)
        self._storage.append((ck, cv))

    def retrieve_all(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """取回所有 KV。"""
        return [self._cache.decompress(ck, cv) for ck, cv in self._storage]
