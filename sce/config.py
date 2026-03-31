"""统一配置：所有超参数集中管理，零硬编码。"""
from dataclasses import dataclass, field
import torch


@dataclass
class ExperimentConfig:
    """
    实验配置。每一个数值都有明确理由，非任意选择。
    通过dataclass + field实现零硬编码。
    """

    # === 硬件 ===
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed: int = 42

    # === 数据 ===
    vocab_size: int = 64        # 足以区分任务差异
    seq_len: int = 16           # 原始规模，先验证向量化 + EWC
    num_tasks: int = 5
    train_samples: int = 5000
    test_samples: int = 1000
    batch_size: int = 64

    # === 模型 ===
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0       # 小数据集禁用dropout

    # === 训练 ===
    lr: float = 1e-3
    steps_per_task: int = 300   # 300步平衡收敛与速度

    # === EWC 参数 ===
    ewc_lambda: float = 400.0          # Fisher正则化强度
    ewc_fisher_samples: int = 200      # 计算Fisher信息的采样数

    # === Replay 参数 ===
    replay_buffer_per_task: int = 500   # 每任务保存的回放样本数（增大以提升回放效果）

    # === SCE 参数 ===
    surprise_ema_alpha: float = 0.05    # Surprise EMA平滑系数
    surprise_sigma: float = 2.0         # 触发阈值 = mean + sigma * std
    surprise_warmup: int = 9999          # 禁止中途 surprise 触发（只在任务边界 grow）
    kd_alpha: float = 0.1              # KD权重（降低以减少新任务学习阻力）
    kd_temperature: float = 2.0        # KD softmax温度

    # === 实验 ===
    num_runs: int = 1                   # 单次运行（加速迭代）
