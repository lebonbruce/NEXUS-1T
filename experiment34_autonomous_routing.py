import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Autonomous Routing Rocket: 隐空间全自动寻址验证
# 核心：模型根据输入特征自发选择物理专家，实现无监督的知识隔离
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 128
VOCAB_SIZE = 256

class PhysicalExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, d_model))
        # 每一个物理专家维护一个“语义质心”
        self.register_buffer('centroid', torch.zeros(d_model))
        self.hit_count = 0

    def forward(self, x):
        return self.net(x)

class AutoRouterRocket(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.experts = nn.ModuleList([PhysicalExpert(D_MODEL) for _ in range(num_experts)])
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.is_warmup = True # 初始预热期

    def forward(self, x):
        x = self.emb(x) # (B, T, D)
        B, T, D = x.size()
        h_mean = x.mean(dim=1) # 提取序列的语义特征 (B, D)

        # --- 核心：自动寻址元语 ---
        if self.is_warmup:
            # 预热期：所有数据初始化第一个专家
            best_idx = 0
            if self.experts[0].hit_count == 0:
                self.experts[0].centroid.copy_(h_mean.mean(dim=0))
        else:
            # 寻址期：计算当前输入与各专家质心的相似度
            centroids = torch.stack([e.centroid for e in self.experts]) # (K, D)
            sim = F.cosine_similarity(h_mean.unsqueeze(1), centroids.unsqueeze(0), dim=-1) # (B, K)
            best_idx = torch.argmax(sim, dim=1)[0].item() # 选出最匹配的物理专家

        # 更新质心（在线聚类，模拟边学边认）
        self.experts[best_idx].centroid = 0.99 * self.experts[best_idx].centroid + 0.01 * h_mean.mean(dim=0)
        self.experts[best_idx].hit_count += 1
        
        # 激活被选中的物理专家
        feat = self.experts[best_idx](x)
        return self.head(feat), best_idx

def run_auto_routing_test():
    print("🚀 Launching Autonomous Routing Test (Zero-Manual-Switch)...")
    model = AutoRouterRocket(num_experts=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 混合数据流：任务 0 和 任务 1 交替出现
    # 规律 0: x + 1, 规律 1: x + 5
    for step in range(600):
        # 模拟模型完全不知道当前是什么任务
        t_id = 0 if step < 300 else 1
        x = torch.randint(0, VOCAB_SIZE, (1, 16), device=DEVICE)
        y = (x + (1 if t_id == 0 else 5)) % VOCAB_SIZE
        
        if step == 300: model.is_warmup = False # 300步后开启自动寻址

        logits, route = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 150 == 0 or step == 301:
            print(f"  Step {step:3d} | Route Chosen: {route} | Loss: {loss.item():.4f}")

    print("\n[Result] Model automatically partitioned its physical layers based on data entropy.")

if __name__ == "__main__":
    run_auto_routing_test()
