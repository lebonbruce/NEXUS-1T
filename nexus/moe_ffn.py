"""
NEXUS 细胞分裂 FFN — 方案 E：冻结旧 + 扰动新

四次迭代的最终方案：
  A: deepcopy+merge     → 合并退化 → BWT=-19%
  B: random+freeze      → 特征失配 → AA 暴跌（Reversal 1.3%）
  D: deepcopy+perturb   → KD target 漂移 → 新任务学不会（Reversal 1.7%）
  E: deepcopy+perturb+freeze → 冻结=稳定KD target, 扰动=打破对称性

根因分析（为什么方案 D 失败）：
  forward_with_kd 中 KD loss = MSE(新expert输出, 旧expert输出)
  方案 D 不冻结旧 expert → replay 时旧 expert 被更新 → KD target 漂移
  → 新 expert 追逐漂移的 target → 无法收敛到新任务的最优解

方案 E 的设计逻辑：
  1. deepcopy + 扰动：保持特征匹配 + 打破对称性
  2. 冻结旧 expert：稳定 KD target + 保护旧知识
  3. EWC 保护共享层：共享层的重要参数不偏移
  4. Replay 更新共享层：replay 梯度只流过冻结 expert（不更新它），但更新共享层

理论预期：
  - AA ≈ 方案 A（deepcopy 保持特征匹配 → 新任务能学会）
  - BWT ≈ 方案 B（冻结 → 旧 expert 不被修改 → 零遗忘）
  - 速度 ≈ 方案 A（相同计算量）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .config import NexusConfig


class SwiGLUExpert(nn.Module):
    """
    SwiGLU FFN Expert — DeepSeek-V3 官方的 MLP 架构。

    SwiGLU(x) = SiLU(W1·x) * (W3·x)
    output = W2 · SwiGLU(x)
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)    # down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)    # up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DynamicMoEFFN(nn.Module):
    """
    细胞分裂 FFN — 方案 E：冻结旧 expert + deepcopy 扰动新 expert。

    生命周期：
      Task 0: expert_0 训练 → 冻结 expert_0
      Task 1: expert_1 = deepcopy(expert_0) + 扰动 → 训练 expert_1 → 冻结 expert_1
      Task 2: expert_2 = deepcopy(expert_1) + 扰动 → 训练 expert_2 → 冻结 expert_2
      ...
      推理: task_id → task_expert_map[task_id] → 对应的冻结 expert
    """

    def __init__(self, config: NexusConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        # 扰动幅度：控制新 expert 与旧 expert 的初始差异
        # 按参数标准差的 1% 缩放，足以打破对称性但不破坏特征匹配
        self.perturbation_scale = 0.01

        # === Expert 池 ===
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.d_model, config.d_ff)
        ])
        self.active_idx = 0

        # task_id → expert_idx 映射
        self.task_expert_map: dict[int, int] = {}

    def grow(self, device: torch.device):
        """
        细胞分裂：冻结旧 expert + deepcopy 扰动新 expert。

        步骤:
          1. 冻结当前 active expert（保护旧知识 + 稳定 KD target）
          2. deepcopy 创建新 expert（保持与共享层的特征匹配）
          3. 添加自适应随机扰动（打破对称性，避免退化合并）
          4. 新 expert 设为可训练（学习新任务）
        """
        # [1] 冻结旧 expert（双重作用：保护旧知识 + 稳定 KD target）
        old_expert = self.experts[self.active_idx]
        for p in old_expert.parameters():
            p.requires_grad = False

        # [2] deepcopy 创建新 expert
        new_expert = copy.deepcopy(old_expert).to(device)

        # [3] 添加自适应随机扰动打破对称性
        # 按参数标准差缩放，保证扰动量级与参数本身匹配
        with torch.no_grad():
            for p in new_expert.parameters():
                param_std = p.data.std()
                noise = torch.randn_like(p.data) * param_std * self.perturbation_scale
                p.data.add_(noise)

        # [4] 新 expert 可训练
        for p in new_expert.parameters():
            p.requires_grad = True

        self.experts.append(new_expert)
        self.active_idx = len(self.experts) - 1

    def register_task(self, task_id: int):
        """记录 task → expert 映射。"""
        self.task_expert_map[task_id] = self.active_idx

    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        """
        前向传播。按 task_id 路由到对应 expert。

        训练时: 路由到 active expert（可训练）
        评估时: 按 task_id 映射到对应的冻结 expert
        """
        if task_id is not None and task_id in self.task_expert_map:
            idx = self.task_expert_map[task_id]
        else:
            idx = self.active_idx
        return self.experts[idx](x)

    def get_old_expert_output(self, x: torch.Tensor):
        """获取上一个 expert 的输出（用于 KD 蒸馏）。
        旧 expert 已冻结 → KD target 稳定 → 新 expert 可以有效学习。"""
        if self.active_idx == 0:
            return None
        with torch.no_grad():
            return self.experts[self.active_idx - 1](x)
