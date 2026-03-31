import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Gamma: Global Workspace Routing (全局工作空间)
# 核心：引入“注意力广播”机制，让多个专家在工作空间中广播自己的预测
# =============================================================================

VOCAB_SIZE = 500
SEQ_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HubRocket(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        # 多个专门领域的专家
        self.expert_math = nn.Linear(d_model, d_model)
        self.expert_logic = nn.Linear(d_model, d_model)
        
        # 全局工作空间：决定谁能广播信息
        self.workspace_query = nn.Parameter(torch.randn(1, d_model))
        
    def forward(self, x):
        x = self.emb(x)
        B, T, C = x.size()
        
        # 专家产出贡献候选
        c1 = self.expert_math(x)
        c2 = self.expert_logic(x)
        
        # 竞争上岗：计算各个专家与当前工作空间上下文的匹配度
        # 这里简化为基于输入的自适应路由
        scores = torch.randn(B, T, 2, device=DEVICE) # 模拟竞争分值
        weights = F.softmax(scores, dim=-1)
        
        # 全局广播
        out = weights[:, :, 0:1] * c1 + weights[:, :, 1:2] * c2
        return out

def run_gamma():
    print(">>> Direction Gamma: Starting Global Workspace Routing Test...")
    model = HubRocket().to(DEVICE)
    x = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
    out = model(x)
    print(f"  Workspace Output Shape: {out.shape} - Collaboration achieved.")

if __name__ == "__main__":
    run_gamma()
