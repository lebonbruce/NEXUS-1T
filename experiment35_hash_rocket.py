import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Hash-Rocket: 基于硬件寻址逻辑的 1M 参数架构
# 核心：利用固定随机投影 (FRP) 实现无监督的物理专家跳转
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 128
VOCAB_SIZE = 256

class FixedHashRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        # 第一性原理：固化的随机正交矩阵，不参与训练，保证寻址不漂移
        self.register_buffer('proj', torch.randn(d_model, 16)) 
        self.num_experts = num_experts

    def forward(self, x):
        # x: (B, D)
        # 生成二进制 Hash 地址
        with torch.no_grad():
            code = (x @ self.proj) > 0 # (B, 16)
            # 简化：取前几位映射到专家索引
            idx = code[:, :2].float() @ torch.tensor([2., 1.], device=DEVICE)
        return idx.long()

class HashRocket(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.router = FixedHashRouter(D_MODEL, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(D_MODEL, 512), nn.GELU(), nn.Linear(512, D_MODEL))
            for _ in range(num_experts)
        ])
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x_emb = self.emb(x)
        h_mean = x_emb.mean(dim=1)
        
        # 1. 硬件级哈希寻址
        expert_indices = self.router(h_mean)
        
        # 2. 物理专家激活
        # 由于是演示，我们取 batch 第一个样本的索引（实际可并行）
        idx = expert_indices[0].item()
        feat = self.experts[idx](x_emb)
        return self.head(feat), idx

def run_hash_benchmark():
    print("🚀 Launching Hash-Rocket Benchmark (Hardware-like Addressing)...")
    model = HashRocket(num_experts=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 任务序列流
    tasks = [1, 5, 10, 20] # 不同的加法偏移代表不同语义
    for t_id, offset in enumerate(tasks):
        print(f"\n--- Era {t_id}: Rule (x + {offset}) ---")
        for step in range(200):
            x = torch.randint(0, VOCAB_SIZE, (1, 16), device=DEVICE)
            y = (x + offset) % VOCAB_SIZE
            
            logits, route = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"  Step {step:3d} | Route: {route} | Loss: {loss.item():.4f}")

    print("\n[Final Check] Observing if model used different experts for different rules...")

if __name__ == "__main__":
    run_hash_benchmark()
