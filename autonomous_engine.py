import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import time
import os

# =============================================================================
# 第一性原理实验环境：1M参数级 自动演化引擎
# 目标：通过“稳态疲劳”与“Delta学习律”彻底超越传统T架构
# =============================================================================

VOCAB_SIZE = 1000
SEQ_LEN = 32
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 高性能流式数据集 (模拟真实世界知识不断涌现) ---
class KnowledgeStream:
    def __init__(self, vocab_size=1000, seq_len=32):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.task_pointer = 0
        
    def next_batch(self, batch_size, task_id=None):
        if task_id is None: task_id = self.task_pointer
        # 每一个Task都有完全不同的数学规律
        # Task 0: 周期性规律
        # Task 1: 镜像/对称规律
        # Task 2: 质数跳跃规律
        # ...
        X = []
        for _ in range(batch_size):
            if task_id % 3 == 0: # 周期
                start = np.random.randint(0, self.vocab_size)
                step = np.random.randint(1, 10)
                seq = [(start + i * step) % self.vocab_size for i in range(self.seq_len + 1)]
            elif task_id % 3 == 1: # 对称
                half = self.seq_len // 2
                first_half = np.random.randint(0, self.vocab_size, size=half).tolist()
                seq = first_half + first_half[::-1] + [np.random.randint(0, 10)]
            else: # 随机偏移
                start = np.random.randint(0, self.vocab_size)
                seq = [(start + i * i) % self.vocab_size for i in range(self.seq_len + 1)]
            X.append(seq)
        
        data = torch.tensor(X, dtype=torch.long, device=DEVICE)
        return data[:, :-1], data[:, 1:]

# --- 2. 热力学底座：标准T-Block (用于1M对照组) ---
class BaselineTransformer(nn.Module):
    def __init__(self, d_model=128, n_layers=6, n_heads=4, hidden_dim=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ln2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model))
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        
    def forward(self, x):
        x = self.emb(x) + self.pos
        for l in self.layers:
            res = x
            x, _ = l['attn'](l['ln1'](x), l['ln1'](x), l['ln1'](x), need_weights=False, 
                             attn_mask=torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), 1).bool())
            x = res + x
            x = x + l['ffn'](l['ln2'](x))
        return self.head(self.ln_f(x))

# --- 3. 火箭架构：Delta-Organic Rocket (DOR) ---
# 核心元语：
# a) Delta-Rule Associative Memory (短期快速适应)
# b) Homeostatic Fatigue Trigger (稳态疲劳触发结构生长)
# c) Topological Growth (在SOM网格中长出新结构)

class OrganicFFN(nn.Module):
    def __init__(self, d_model, initial_dim=256):
        super().__init__()
        self.d_model = d_model
        # 稳态记忆池
        self.experts = nn.ModuleList([nn.Linear(d_model, initial_dim)])
        self.out_projs = nn.ModuleList([nn.Linear(initial_dim, d_model)])
        self.usage = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.active_expert = 0
        self.fatigue_limit = 500000 # 经过50万个token后，该结构进入“稳态固化”
        
    def grow(self):
        new_dim = self.experts[0].out_features
        # 冻结旧专家
        for p in self.experts[-1].parameters(): p.requires_grad = False
        for p in self.out_projs[-1].parameters(): p.requires_grad = False
        
        # 长出新专家
        self.experts.append(nn.Linear(self.d_model, new_dim).to(DEVICE))
        self.out_projs.append(nn.Linear(new_dim, self.d_model).to(DEVICE))
        self.active_expert = len(self.experts) - 1
        self.usage.data = torch.cat([self.usage.data, torch.zeros(1, device=DEVICE)])
        print(f"  [DOR] Structure Branching: Stage {len(self.experts)} activated.")

    def forward(self, x):
        # 简单路由：总是使用最新的“新鲜”专家进行学习，旧专家作为“固定记忆”
        # 在更复杂的版本中，我们会使用SOM路由
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        # 统计活跃度
        if self.training:
            self.usage.data[self.active_expert] += x_flat.size(0)
            if self.usage.data[self.active_expert] > self.fatigue_limit:
                self.grow()
        
        # 计算：当前只激活最顶层的专家（为了1M算力对比的公平性）
        # 实际上可以通过Residual连接融合旧专家的输出
        mid = F.gelu(self.experts[self.active_expert](x_flat))
        out = self.out_projs[self.active_expert](mid)
        return out.view(B, T, C)

class DeltaAttention(nn.Module):
    """ 实现Delta学习律的注意力元语：在推理中学习 """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.beta = nn.Parameter(torch.ones(n_heads) * 0.5)
        
    def forward(self, x):
        B, T, C = x.size()
        H, D = self.head_dim, self.n_heads
        q = self.q_proj(x).view(B, T, D, H)
        k = F.normalize(self.k_proj(x).view(B, T, D, H), p=2, dim=-1)
        v = self.v_proj(x).view(B, T, D, H)
        
        # 这里的 thermodynamic law：
        # 我们不使用传统的Softmax Attention，而是使用线性Delta注意力
        # 它允许我们在Sequence维度上通过Recurrence积累“快速权重”
        state = torch.zeros(B, D, H, H, device=x.device)
        outputs = []
        for t in range(T):
            qt = q[:, t].unsqueeze(2) # (B, D, 1, H)
            kt = k[:, t].unsqueeze(3) # (B, D, H, 1)
            vt = v[:, t].unsqueeze(3) # (B, D, H, 1)
            
            # 检索
            yt = torch.matmul(state, qt.transpose(-2, -1)).transpose(-2, -1)
            outputs.append(yt.squeeze(2))
            
            # Delta更新：学习误差
            err = vt - torch.matmul(state, kt)
            state = state + torch.sigmoid(self.beta).view(1, D, 1, 1) * torch.matmul(err, kt.transpose(-2, -1))
            
        return torch.stack(outputs, dim=1).reshape(B, T, C)

class OrganicRocket(nn.Module):
    def __init__(self, d_model=128, n_layers=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': DeltaAttention(d_model, 4),
                'ln2': nn.LayerNorm(d_model),
                'ffn': OrganicFFN(d_model, initial_dim=512)
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        
    def forward(self, x):
        x = self.emb(x) + self.pos
        for l in self.layers:
            x = x + l['attn'](l['ln1'](x))
            x = x + l['ffn'](l['ln2'](x))
        return self.head(self.ln_f(x))

# --- 4. 自动化实验引擎 ---
def run_autonomous_experiment():
    print(f"🚀 Initializing 1M-Param Autonomous Engine on {DEVICE}...")
    stream = KnowledgeStream()
    
    # 初始化模型
    model_base = BaselineTransformer().to(DEVICE)
    model_rocket = OrganicRocket().to(DEVICE)
    
    # 统计参数
    base_params = sum(p.numel() for p in model_base.parameters())
    rocket_params = sum(p.numel() for p in model_rocket.parameters() if p.requires_grad)
    print(f"  Baseline Params: {base_params/1e6:.2f}M")
    print(f"  Rocket Active Params: {rocket_params/1e6:.2f}M")
    
    # 连续学习评估闭环
    history = {"base": [], "rocket": []}
    
    # 我们运行10个连续的知识任务
    for task_id in range(10):
        print(f"\n--- Current Knowledge Era: {task_id} ---")
        
        # 训练
        opt_base = optim.Adam(model_base.parameters(), lr=1e-3)
        opt_rocket = optim.Adam(model_rocket.parameters(), lr=1e-3)
        
        for step in range(500):
            bx, by = stream.next_batch(BATCH_SIZE, task_id)
            
            # Baseline Update
            opt_base.zero_grad()
            l_base = F.cross_entropy(model_base(bx).reshape(-1, VOCAB_SIZE), by.reshape(-1))
            l_base.backward()
            opt_base.step()
            
            # Rocket Update
            opt_rocket.zero_grad()
            l_rocket = F.cross_entropy(model_rocket(bx).reshape(-1, VOCAB_SIZE), by.reshape(-1))
            l_rocket.backward()
            opt_rocket.step()
            
            if step % 100 == 0:
                print(f"  Step {step:3d} | Base Loss: {l_base.item():.4f} | Rocket Loss: {l_rocket.item():.4f}")
        
        # 遗忘率测试 (测试对所有历史知识的保留)
        print(f"  [Evaluation] Testing all previous eras...")
        base_forget = 0; rocket_forget = 0
        for old_id in range(task_id + 1):
            ex, ey = stream.next_batch(100, old_id)
            with torch.no_grad():
                lb = F.cross_entropy(model_base(ex).reshape(-1, VOCAB_SIZE), ey.reshape(-1)).item()
                lr = F.cross_entropy(model_rocket(ex).reshape(-1, VOCAB_SIZE), ey.reshape(-1)).item()
                base_forget += lb
                rocket_forget += lr
        
        avg_b = base_forget / (task_id + 1)
        avg_r = rocket_forget / (task_id + 1)
        print(f"  >>> Era {task_id} Cumulative Loss | Base: {avg_b:.4f} | Rocket: {avg_r:.4f}")
        print(f"  >>> Improvement: {avg_b/avg_r:.2f}x better")

if __name__ == "__main__":
    run_autonomous_experiment()
