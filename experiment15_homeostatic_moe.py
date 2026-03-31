import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

VOCAB_SIZE = 100
SEQ_LEN = 16
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_task_data(task_id, num_samples):
    X, Y = [], []
    for _ in range(num_samples):
        if task_id == 0:
            start = np.random.randint(0, 50) * 2
            seq = [(start + i * 2) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        elif task_id == 1:
            start = np.random.randint(0, 50) * 2 + 1
            seq = [(start + i * 2) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        else:
            start = np.random.randint(0, VOCAB_SIZE)
            seq = [(start + i * 5) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        X.append(seq[:-1]); Y.append(seq[1:])
    return torch.tensor(X, dtype=torch.long, device=DEVICE), torch.tensor(Y, dtype=torch.long, device=DEVICE)

# --- Homeostatic MoE Layer ---
class HomeostaticMoELayer(nn.Module):
    def __init__(self, d_model, max_experts=16, hidden_dim=256):
        super().__init__()
        self.d_model = d_model
        self.max_experts = max_experts
        self.hidden_dim = hidden_dim
        
        # We start with 1 active expert
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )])
        
        # Router
        self.router = nn.Linear(d_model, 1)
        
        # Homeostatic state: Accumulated usage (fatigue)
        self.register_buffer('usage', torch.zeros(1))
        self.fatigue_threshold = 5000.0 # Total tokens processed by this expert before it gets "tired"
        
    def add_expert(self):
        new_expert = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.d_model)
        ).to(DEVICE)
        self.experts.append(new_expert)
        
        # Expand router
        old_router = self.router
        self.router = nn.Linear(self.d_model, len(self.experts)).to(DEVICE)
        with torch.no_grad():
            self.router.weight[:-1] = old_router.weight
            self.router.bias[:-1] = old_router.bias
            nn.init.normal_(self.router.weight[-1], std=0.02)
            
        # Add usage tracker
        new_usage = torch.zeros(len(self.experts), device=DEVICE)
        new_usage[:-1] = self.usage
        self.usage = new_usage
        print(f"[Homeostatic] Added Expert {len(self.experts)}. Usage states: {self.usage}")

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        # 1. Routing with Fatigue Penalty
        logits = self.router(x_flat) # (N, num_experts)
        
        # Apply fatigue penalty: reduce logits for over-used experts
        # penalty = usage / scale
        penalty = self.usage.unsqueeze(0) * 0.001
        logits = logits - penalty
        
        probs = F.softmax(logits, dim=-1)
        _, best_idx = torch.max(probs, dim=-1)
        
        # 2. Update usage (Homeostasis)
        if self.training:
            for i in range(len(self.experts)):
                self.usage[i] += (best_idx == i).sum().float()
                
            # Trigger growth if the current newest expert is also becoming tired
            if self.usage[-1] > self.fatigue_threshold and len(self.experts) < self.max_experts:
                self.add_expert()
        
        final_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (best_idx == i)
            if mask.any():
                final_out[mask] = expert(x_flat[mask])
                
        return final_out.view(B, T, C), best_idx

class HomeoRocket(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = HomeostaticMoELayer(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.emb(x) + self.pos
        res = x
        x, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False, 
                         attn_mask=torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=1).bool())
        x = res + x
        res = x
        moe_out, _ = self.moe(self.ln2(x))
        x = res + moe_out
        x = self.ln_f(x)
        return self.head(x)

def train_homeo(model, x, y):
    # For homeostasis, we use a single optimizer but growth happens dynamically
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(10):
        for i in range(0, len(x), BATCH_SIZE):
            bx, by = x[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]
            
            # Re-init optimizer if model size changed (expert added)
            # In a real system, we'd use an optimizer that supports param groups
            if len(list(model.parameters())) != len(optimizer.param_groups[0]['params']):
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), by.view(-1))
            loss.backward()
            optimizer.step()

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()

if __name__ == "__main__":
    print(f"\n{'='*50}\nRunning Homeo-Rocket (Dynamic Fatigue Growth)\n{'='*50}")
    model = HomeoRocket(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        train_homeo(model, x, y)
        for eval_id, (tx, ty) in enumerate(test_tasks):
            loss = evaluate(model, tx, ty)
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")
