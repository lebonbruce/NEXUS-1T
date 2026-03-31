"""
Surprise-Driven Expert Growth: 基于预测误差统计的自动结构生长。

与V21的Homeostatic Fatigue的区别：
- V21: 硬编码计数器触发（处理5000个token后生长）
- 本实现: 基于prediction loss的EMA统计量检测domain shift

原理：当loss突然显著偏离历史均值（超过sigma个标准差），
说明当前输入与模型已学知识存在根本不同 → 触发结构生长。
"""
import torch
import torch.nn as nn
import math


class SurpriseMonitor:
    """
    基于EMA的surprise检测器。
    追踪loss的均值和方差，当loss突然飙升时触发alarm。
    """

    def __init__(self, alpha: float = 0.05, sigma: float = 2.0,
                 warmup: int = 50):
        self.alpha = alpha       # EMA平滑系数
        self.sigma = sigma       # 触发阈值的标准差倍数
        self.warmup = warmup     # 预热期（期间不触发）
        self._mean = 0.0
        self._var = 0.0
        self._step = 0
        self._triggered = False

    def update(self, loss_value: float) -> bool:
        """
        更新统计量，返回是否检测到surprise。
        """
        self._step += 1

        if self._step <= self.warmup:
            # 预热期：只积累统计量
            if self._step == 1:
                self._mean = loss_value
                self._var = 0.0
            else:
                self._mean = self.alpha * loss_value + (1 - self.alpha) * self._mean
                diff = loss_value - self._mean
                self._var = self.alpha * (diff ** 2) + (1 - self.alpha) * self._var
            self._triggered = False
            return False

        # 计算当前阈值
        std = math.sqrt(self._var + 1e-8)
        threshold = self._mean + self.sigma * std

        # 检查surprise
        is_surprised = loss_value > threshold and not self._triggered

        # 更新EMA
        self._mean = self.alpha * loss_value + (1 - self.alpha) * self._mean
        diff = loss_value - self._mean
        self._var = self.alpha * (diff ** 2) + (1 - self.alpha) * self._var

        if is_surprised:
            self._triggered = True
        return is_surprised

    def reset_trigger(self):
        """在结构生长完成后重置触发状态，允许未来再次触发。"""
        self._triggered = False
        self._step = 0  # 重新进入预热期


class SurpriseFFN(nn.Module):
    """
    带surprise-driven生长的FFN层。
    初始只有1个Expert，当surprise触发时分裂出新Expert。
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Expert池
        self.experts = nn.ModuleList([self._make_expert()])
        self.active_idx = 0

        # task_id → expert_idx 的映射
        self.task_expert_map: dict[int, int] = {}

    def _make_expert(self) -> nn.Sequential:
        """创建一个FFN Expert。"""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.d_model),
        )

    def grow(self, device: torch.device):
        """
        冻结当前active expert，创建新expert。
        """
        # 冻结当前Expert
        for p in self.experts[self.active_idx].parameters():
            p.requires_grad = False

        # 创建新Expert
        new_expert = self._make_expert().to(device)
        self.experts.append(new_expert)
        self.active_idx = len(self.experts) - 1

    def register_task(self, task_id: int):
        """记录当前task使用哪个expert（用于eval时路由）。"""
        self.task_expert_map[task_id] = self.active_idx

    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        # eval时：根据task_id路由到对应expert
        if not self.training and task_id is not None:
            idx = self.task_expert_map.get(task_id, self.active_idx)
        else:
            # train时：始终使用active expert
            idx = self.active_idx
        return self.experts[idx](x)

    def get_old_expert_output(self, x: torch.Tensor) -> torch.Tensor:
        """获取上一个Expert的输出（用于KD）。"""
        if self.active_idx == 0:
            return None
        with torch.no_grad():
            return self.experts[self.active_idx - 1](x)
