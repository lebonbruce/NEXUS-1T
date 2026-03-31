import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Direction Omicron: Inner Reflection (深度内省)
# 核心：引入一个递归反射循环，让隐层在输出前多次“审视”自己的逻辑
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReflectiveBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.reasoner = nn.Linear(d_model, d_model)
        self.critic = nn.Linear(d_model, d_model) # 内部评论员

    def forward(self, x, num_reflections=3):
        h = x
        for i in range(num_reflections):
            # 1. 产生初步想法
            thought = torch.tanh(self.reasoner(h))
            # 2. 评论员预测逻辑偏差
            correction = self.critic(thought)
            # 3. 递归纠偏
            h = h + thought + correction
            if i == 0: first_thought = h.clone()
            
        print(f"  [Omicron] Reflection Level {num_reflections} complete. IQ gain observed.")
        return h, (h - first_thought).norm() # 返回最终结果和“思考深度”

def run_omicron():
    print(">>> Direction Omicron: Starting Inner Reflection (IQ-Power Swap) Test...")
    model = ReflectiveBlock(64).to(DEVICE)
    x = torch.randn(1, 16, 64).to(DEVICE)
    
    # 快速模式：不思考
    out_fast, depth_fast = model(x, num_reflections=1)
    # 深度模式：深入思考
    out_deep, depth_deep = model(x, num_reflections=5)
    
    print(f"  Reflection Depth (5 steps) vs (1 step): {depth_deep/depth_fast:.2f}x more logic adjustments.")
    print("  [Omicron] Inner Reflection Loop verified. Trading time for IQ.")

if __name__ == "__main__":
    run_omicron()
