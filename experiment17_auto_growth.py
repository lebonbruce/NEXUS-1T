import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# =============================================================================
# Alpha: Autonomous Emergence (自主结构涌现)
# 核心：利用认知失调（Entropy）作为物理生长信号
# =============================================================================

VOCAB_SIZE = 500
SEQ_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, d_model))
        self.head = nn.Linear(d_model, VOCAB_SIZE)
    def forward(self, x):
        return self.head(self.net(x))

class EmergenceRocket(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.experts = nn.ModuleList([AutoExpert(d_model)])
        self.growth_threshold = 2.5 # 认知失调阈值
        self.surprise_buffer = []

    def check_and_grow(self, current_loss):
        self.surprise_buffer.append(current_loss)
        if len(self.surprise_buffer) > 20:
            avg_surprise = sum(self.surprise_buffer[-20:]) / 20
            if avg_surprise > self.growth_threshold:
                # 触发分裂：冻结旧突触，长出新路径
                for p in self.experts.parameters(): p.requires_grad = False
                self.experts.append(AutoExpert(self.d_model).to(DEVICE))
                self.surprise_buffer = []
                print(f"\n[Alpha] Cognitive Dissonance detected ({avg_surprise:.4f}). Branching to Expert {len(self.experts)-1}...")
                return True
        return False

    def forward(self, x, expert_idx=None):
        if expert_idx is None: expert_idx = len(self.experts) - 1
        x = self.emb(x)
        return self.experts[expert_idx](x)

def run_alpha():
    print(">>> Direction Alpha: Starting Autonomous Emergence Test...")
    model = EmergenceRocket().to(DEVICE)
    optimizer = optim.Adam(model.experts[-1].parameters(), lr=5e-3)
    
    # 模拟一个不断变化的知识流
    for step in range(1000):
        # 动态改变规律：每 300 步切换一次数学规律
        task_id = step // 300
        start = np.random.randint(0, VOCAB_SIZE)
        if task_id == 0: seq = [(start + i * 2) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        elif task_id == 1: seq = [(start + i * 5) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        else: seq = [(start + i * i) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        
        data = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        x, y = data[:, :-1], data[:, 1:]
        
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 自主决定是否生长
        if model.check_and_grow(loss.item()):
            # 热更新优化器：只训练新专家
            optimizer = optim.Adam(model.experts[-1].parameters(), lr=5e-3)
            
        if step % 100 == 0:
            print(f"  Step {step:4d} | Loss: {loss.item():.4f} | Experts: {len(model.experts)}")

if __name__ == "__main__":
    run_alpha()
