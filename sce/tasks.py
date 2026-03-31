"""
Benchmark任务定义：5个有本质区别的算法推理序列任务。
每个任务是 seq2seq 映射：输入序列 → 输出序列。
任务间需要根本不同的计算能力，确保区分度。
"""
import torch

# 任务名称（用于报告）
TASK_NAMES = [
    "LinearMap",      # y_i = (2*x_i + 3) % V
    "Reversal",       # y = reverse(x)
    "CumulativeSum",  # y_i = cumsum(x)_i % V
    "Sort",           # y = sort(x)
    "ParityEncode",   # y_i = x_i % primes[i]
]

# Parity任务使用的质数列表（支持 seq_len 最大 128）
_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
    137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
    313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613,
    617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719,
]


def generate_task_data(task_id: int, num_samples: int,
                       vocab_size: int, seq_len: int) -> tuple:
    """
    生成指定任务的输入-输出对。

    Args:
        task_id: 任务编号 (0-4)
        num_samples: 样本数量
        vocab_size: 词表大小
        seq_len: 序列长度

    Returns:
        (X, Y): 都是 shape (num_samples, seq_len) 的 LongTensor
    """
    if task_id == 0:
        # LinearMap: y_i = (2 * x_i + 3) % vocab_size
        # 需要的能力：逐位置线性变换
        X = torch.randint(0, vocab_size, (num_samples, seq_len))
        Y = (2 * X + 3) % vocab_size

    elif task_id == 1:
        # Reversal: y = reverse(x)
        # 需要的能力：全局位置重映射
        X = torch.randint(0, vocab_size, (num_samples, seq_len))
        Y = X.flip(dims=[1])

    elif task_id == 2:
        # CumulativeSum: y_i = (x_1 + x_2 + ... + x_i) % vocab_size
        # 需要的能力：记忆累积状态
        # 使用较小值域防止前缀和太快饱和
        X = torch.randint(0, vocab_size // 4, (num_samples, seq_len))
        Y = torch.cumsum(X, dim=1) % vocab_size

    elif task_id == 3:
        # Sort: y = sort(x)
        # 需要的能力：全局比较和排序
        X = torch.randint(0, vocab_size, (num_samples, seq_len))
        Y, _ = torch.sort(X, dim=1)

    elif task_id == 4:
        # ParityEncode: y_i = x_i % primes[i]
        # 需要的能力：位置相关的不同模运算
        X = torch.randint(0, vocab_size, (num_samples, seq_len))
        primes = torch.tensor(_PRIMES[:seq_len], dtype=torch.long)
        Y = X % primes.unsqueeze(0)  # 广播: (N, T) % (1, T)

    else:
        raise ValueError(f"未知任务ID: {task_id}，支持范围 0-4")

    return X, Y
