"""
双向Delta Rule Attention: 基于Schmidhuber (1992) Fast Weights的线性关联存储。
通过前向+反向两次扫描实现双向上下文访问。

与标准Self-Attention的本质区别：
- Self-Attention: 每次查询都重新计算与所有位置的相似度（无状态）
- Delta Attention: 通过Rank-1 update累积"快速权重"矩阵（有状态、递归）

理论根基: Schlag et al. 2021 "Linear Transformers Are Secretly Fast Weight Programmers"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaRuleAttention(nn.Module):
    """
    单方向Delta Rule Attention。
    在序列上递归地构建关联记忆矩阵，实现"边看边学"。
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # 可学习的更新速率（每个Head独立控制学习强度）
        self.beta = nn.Parameter(torch.ones(n_heads) * 0.5)

    def _sequential_delta(self, q, k, v):
        """
        顺序Delta Rule扫描。
        q, k, v: (B, H, T, D)
        返回: (B, H, T, D)
        """
        B, H, T, D = q.size()
        # 初始化关联记忆矩阵
        state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        beta = torch.sigmoid(self.beta).view(1, H, 1, 1)

        outputs = []
        for t in range(T):
            qt = q[:, :, t:t+1, :]      # (B, H, 1, D)
            kt = k[:, :, t:t+1, :]      # (B, H, 1, D)
            vt = v[:, :, t:t+1, :]      # (B, H, 1, D)

            # 检索：y_t = State @ q_t^T → (B, H, D, D) @ (B, H, D, 1) → (B, H, D, 1)
            yt = torch.matmul(state, qt.transpose(-2, -1))  # (B, H, D, 1)
            outputs.append(yt.transpose(-2, -1))              # (B, H, 1, D)

            # Delta更新：State += beta * (v_t - State @ k_t) @ k_t^T
            prediction = torch.matmul(state, kt.transpose(-2, -1))  # (B, H, D, 1)
            error = vt.transpose(-2, -1) - prediction                # (B, H, D, 1)
            state = state + beta * torch.matmul(error, kt)            # (B, H, D, D)

        return torch.cat(outputs, dim=2)  # (B, H, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)     # (B, H, T, D)
        k = F.normalize(self.k_proj(x).view(B, T, H, D).transpose(1, 2), dim=-1)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # 双向扫描：前向 + 反向
        y_fwd = self._sequential_delta(q, k, v)
        y_bwd = self._sequential_delta(
            q.flip(2), k.flip(2), v.flip(2)
        ).flip(2)

        # 融合两个方向的结果
        y = (y_fwd + y_bwd) / 2.0
        y = y.transpose(1, 2).reshape(B, T, C)

        return self.o_proj(y)
