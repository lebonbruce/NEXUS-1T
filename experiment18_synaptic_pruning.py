import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Beta: Synaptic Consolidation (突触固化与整合)
# 核心：模拟睡眠机制，将相似专家合并，实现“公理化”
# =============================================================================

VOCAB_SIZE = 500
SEQ_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PruningRocket(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        # 模拟已经长出了 4 个冗余专家
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(4)
        ])
        
    def sleep_and_consolidate(self):
        print("\n[Beta] System entering 'Sleep' mode. Consolidating experts...")
        # 第一性原理：相似即冗余。计算专家权重之间的余弦相似度
        with torch.no_grad():
            w0 = self.experts[0].weight
            w1 = self.experts[1].weight
            sim = F.cosine_similarity(w0.view(-1), w1.view(-1), dim=0)
            print(f"  Expert 0 & 1 Similarity: {sim.item():.4f}")
            
            if sim > 0.8:
                print("  Merging highly redundant experts into a single Axiom...")
                # 简单的均值合并 (更复杂可以用 SVD 提取主成分)
                self.experts[0].weight.data = (w0 + w1) / 2
                # 移除冗余专家 (由于是演示，我们只在逻辑上移除)
                print("  Synaptic pruning complete. Model complexity reduced.")

    def forward(self, x):
        x = self.emb(x)
        # 简单相加模拟专家协作
        out = 0
        for e in self.experts: out += e(x)
        return out

def run_beta():
    print(">>> Direction Beta: Starting Synaptic Consolidation Test...")
    model = PruningRocket().to(DEVICE)
    # 模拟学习后的相似状态
    with torch.no_grad():
        model.experts[1].weight.data = model.experts[0].weight.data + torch.randn_like(model.experts[0].weight) * 0.01
    
    model.sleep_and_consolidate()

if __name__ == "__main__":
    run_beta()
