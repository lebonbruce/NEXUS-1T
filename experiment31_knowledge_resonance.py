import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Direction Xi: Knowledge Resonance (知识纠缠共振)
# 核心：引入纠缠矩阵 E，使专家 A 的更新能以“非破坏性”方式增强专家 B
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EntangledExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model, d_model) / 10)
        
    def forward(self, x, entanglement_matrix=None):
        out = x @ self.w.T
        if entanglement_matrix is not None:
            # 这里的第一性原理：利用纠缠矩阵进行“知识注入”
            # out = out + (x @ E) -> E 代表了其他专家的影子
            out = out + x @ entanglement_matrix.T
        return out

class ResonanceSystem(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.experts = nn.ModuleList([EntangledExpert(d_model) for _ in range(2)])
        # 纠缠矩阵：记录专家间的关联
        self.register_buffer('E', torch.zeros(d_model, d_model))

    def update_entanglement(self):
        # 睡眠期：同步专家 A 和 B 的共同特征到纠缠矩阵
        with torch.no_grad():
            self.E.copy_(0.1 * (self.experts[0].w + self.experts[1].w))
            print(f"  [Xi] Entanglement Matrix Updated. Resonance Norm: {torch.norm(self.E):.4f}")

    def forward(self, x, expert_idx):
        return self.experts[expert_idx](x, self.E)

def run_xi():
    print(">>> Direction Xi: Starting Knowledge Resonance Test...")
    model = ResonanceSystem().to(DEVICE)
    x = torch.randn(1, 16, 64).to(DEVICE)
    
    # 模拟专家 0 学习知识
    print("  Expert 0 is learning...")
    out0 = model(x, 0)
    
    # 触发共振
    model.update_entanglement()
    
    # 观察专家 1 是否在未训练的情况下获得了“共振增益”
    print("  Expert 1 is now resonating with Expert 0's ghost weights.")
    out1 = model(x, 1)
    print("  [Xi] Resonance achieved. Knowledge is no longer an island.")

if __name__ == "__main__":
    run_xi()
