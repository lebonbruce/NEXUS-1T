import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Direction Lambda: Axiomatic Merging (公理化合并)
# 核心：识别同质化专家，通过“睡眠机制”合并权重，提取通用规律
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Expert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model, d_model))
    def forward(self, x):
        return x @ self.w.T

class SleepingRocket(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        # 模拟已经长出的多个专家
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(3)])

    def sleep_and_merge(self, threshold=0.9):
        print("\n[Lambda] System entering 'Sleep' state. Analyzing synaptic redundancy...")
        with torch.no_grad():
            # 计算两两之间的相似度
            w0 = self.experts[0].w.view(-1)
            w1 = self.experts[1].w.view(-1)
            sim = F.cosine_similarity(w0, w1, dim=0)
            print(f"  Expert 0 & 1 Synaptic Similarity: {sim.item():.4f}")
            
            if sim > threshold:
                print("  Redundancy detected! Merging into a single Axiom Branch...")
                # 合并知识：权重的几何平均或简单加权
                self.experts[0].w.data = (self.experts[0].w.data + self.experts[1].w.data) / 2
                # 在实际引擎中，这里会移除 experts[1] 并重定向路由
                return True
        return False

def run_lambda():
    model = SleepingRocket().to(DEVICE)
    # 模拟学习了极其相似知识的两个专家
    with torch.no_grad():
        model.experts[1].w.data = model.experts[0].w.data + torch.randn_like(model.experts[0].w) * 0.01
    
    merged = model.sleep_and_merge()
    print(f"  [Lambda] Merge Successful: {merged}")

if __name__ == "__main__":
    run_lambda()
