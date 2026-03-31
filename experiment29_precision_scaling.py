import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Direction Mu: Precision Scaling (动态秩增长)
# 核心：不再增加块，而是增加现有层的“秩”，用最小算力成本捕捉更复杂的规律
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DynamicRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.base.weight.requires_grad = False # 基础模型冻结
        
        # 动态秩适配器
        self.ranks_a = nn.ParameterList([nn.Parameter(torch.randn(in_dim, 4))])
        self.ranks_b = nn.ParameterList([nn.Parameter(torch.zeros(4, out_dim))])

    def grow_rank(self, increment=4):
        # 增加新的低秩通道
        in_dim = self.base.in_features
        out_dim = self.base.out_features
        device = self.base.weight.device
        self.ranks_a.append(nn.Parameter(torch.randn(in_dim, increment, device=device)))
        self.ranks_b.append(nn.Parameter(torch.zeros(increment, out_dim, device=device)))
        print(f"  [Mu] Rank Expansion: Current total rank = {len(self.ranks_a) * increment}")

    def forward(self, x):
        out = self.base(x)
        # 累加所有秩的贡献
        for a, b in zip(self.ranks_a, self.ranks_b):
            out = out + (x @ a) @ b
        return out

def run_mu():
    print(">>> Direction Mu: Starting Dynamic Rank Scaling Test...")
    model = DynamicRankLinear(64, 64).to(DEVICE)
    x = torch.randn(1, 64).to(DEVICE)
    
    # 模拟在学习过程中复杂度增加
    print("  Learning simple logic...")
    out1 = model(x)
    
    print("  Complexity threshold reached. Growing precision...")
    model.grow_rank()
    out2 = model(x)
    
    print(f"  [Mu] Forward pass remains seamless. Adaptation depth increased.")

if __name__ == "__main__":
    run_mu()
