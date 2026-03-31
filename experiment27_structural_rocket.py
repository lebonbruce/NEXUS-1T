import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# 第一性原理：Structural-Growth Rocket (最终解思路)
# 核心：100% 物理隔离知识，通过“认知失调”驱动结构生长
# =============================================================================

VOCAB_SIZE = 1000
SEQ_LEN = 16
D_MODEL = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_task_data(task_id, batch_size=64):
    X = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=DEVICE)
    if task_id == 0: Y = (X + 1) % VOCAB_SIZE
    else: Y = (X * 7 + 13) % VOCAB_SIZE # 极其不同的数学逻辑
    return X, Y

class GrowthFFN(nn.Module):
    """
    真正的火箭：不再纠结于权重更新，而是直接生长结构。
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 初始只有一个通用专家
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model)])
        self.active_expert = 0

    def spawn_expert(self):
        # 物理生长：长出一个新的隔离分支
        # 冻结旧专家，保护记忆
        for p in self.experts[-1].parameters(): p.requires_grad = False
        self.experts.append(nn.Linear(self.d_model, self.d_model).to(DEVICE))
        self.active_expert = len(self.experts) - 1
        print(f"  [Rocket] Structural Branching: Expert {self.active_expert} is born.")

    def forward(self, x, expert_idx=None):
        if expert_idx is None: expert_idx = self.active_expert
        return self.experts[expert_idx](x)

class StructuralRocket(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.ffn = GrowthFFN(d_model)
        self.head = nn.ModuleList([nn.Linear(d_model, VOCAB_SIZE)])

    def forward(self, x, era=None):
        x = self.emb(x)
        if era is None: era = self.ffn.active_expert
        feat = self.ffn(x, expert_idx=era)
        return self.head[era](feat)

    def evolve(self):
        self.ffn.spawn_expert()
        # 对应的 Head 也要生长
        self.head.append(nn.Linear(self.ffn.d_model, VOCAB_SIZE).to(DEVICE))

def run_rocket_benchmark():
    print(f"🚀 Launching Structural-Growth Rocket...")
    
    m_static = nn.Sequential(nn.Embedding(VOCAB_SIZE, D_MODEL), nn.Linear(D_MODEL, D_MODEL), nn.Linear(D_MODEL, VOCAB_SIZE)).to(DEVICE)
    m_rocket = StructuralRocket(D_MODEL).to(DEVICE)

    for t_id in [0, 1]:
        print(f"\n--- Era: Learning Task {t_id} ---")
        if t_id > 0: m_rocket.evolve()
        
        opt_s = torch.optim.Adam(m_static.parameters(), lr=5e-3)
        # Rocket 只训练当前处于“生长期”的结构
        params_r = list(m_rocket.ffn.experts[-1].parameters()) + list(m_rocket.head[-1].parameters())
        opt_r = torch.optim.Adam(params_r, lr=5e-3)
        
        for step in range(300):
            x, y = get_task_data(t_id)
            
            # Static Model
            opt_s.zero_grad()
            l_s = F.cross_entropy(m_static(x).reshape(-1, VOCAB_SIZE), y.reshape(-1))
            l_s.backward(); opt_s.step()
            
            # Rocket Model
            opt_r.zero_grad()
            l_r = F.cross_entropy(m_rocket(x, era=t_id).reshape(-1, VOCAB_SIZE), y.reshape(-1))
            l_r.backward(); opt_r.step()
            
            if step % 150 == 0: print(f"  Step {step:3d} | Static Loss: {l_s.item():.4f} | Rocket Loss: {l_r.item():.4f}")

    # 评估
    print("\n" + "="*50)
    print("FINAL FIRST-PRINCIPLE REPORT")
    print("="*50)
    with torch.no_grad():
        x0, y0 = get_task_data(0, batch_size=200)
        l0_s = F.cross_entropy(m_static(x0).reshape(-1, VOCAB_SIZE), y0.reshape(-1)).item()
        l0_r = F.cross_entropy(m_rocket(x0, era=0).reshape(-1, VOCAB_SIZE), y0.reshape(-1)).item()
        
    print(f"Task 0 Loss (Traditional): {l0_s:.4f}")
    print(f"Task 0 Loss (Structural Rocket): {l0_r:.4f}")
    print(f"Memory Stability Improvement: {l0_s/l0_r:.2f}x better")

if __name__ == "__main__":
    run_rocket_benchmark()
