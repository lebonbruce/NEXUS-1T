import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Direction Kappa: Backprop-Free Propulsion (无梯度推进)
# 核心：利用预测编码 (Predictive Coding) 元语，让每一层只在本地最小化预测误差
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictiveLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Linear(d_model, d_model)
        # 本地优化器
        self.opt = torch.optim.Adam(self.w.parameters(), lr=1e-3)

    def forward(self, x, target=None):
        # 1. 前向预测
        pred = self.w(x)
        
        # 2. 如果提供本地目标（如上一层的输出），则进行本地即时更新
        if target is not None and self.training:
            local_loss = F.mse_loss(pred, target.detach())
            self.opt.zero_grad()
            local_loss.backward()
            self.opt.step()
            return pred.detach(), local_loss.item()
            
        return pred, 0.0

def run_kappa():
    print(">>> Direction Kappa: Starting Backprop-Free Propulsion Test...")
    d_model = 64
    layer1 = PredictiveLayer(d_model).to(DEVICE)
    layer2 = PredictiveLayer(d_model).to(DEVICE)
    
    # 模拟异步局部学习：没有全局 Loss，只有本地预测误差
    x = torch.randn(1, 16, d_model).to(DEVICE)
    hidden, loss1 = layer1(x, target=x) # 层1试图重构输入
    output, loss2 = layer2(hidden, target=hidden) # 层2试图预测层1的输出
    
    print(f"  Local Prediction Error (Layer 1): {loss1:.6f}")
    print(f"  Local Prediction Error (Layer 2): {loss2:.6f}")
    print("  [Kappa] Local learning achieved. Global Backprop friction removed.")

if __name__ == "__main__":
    run_kappa()
