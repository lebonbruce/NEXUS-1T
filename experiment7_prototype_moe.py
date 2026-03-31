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

class StructuralExpert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
    def forward(self, x):
        return self.net(x)

# --- Prototype-based Routing Layer ---
class PrototypeMoELayer(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleList([])
        self.register_buffer('prototypes', torch.empty(0, d_model))
        
    def add_expert(self, initial_data_mean):
        # initial_data_mean: (d_model,)
        new_expert = StructuralExpert(self.d_model, self.hidden_dim).to(DEVICE)
        self.experts.append(new_expert)
        
        # Add prototype
        self.prototypes = torch.cat([self.prototypes, initial_data_mean.unsqueeze(0)], dim=0)
        print(f"[PrototypeMoE] Added Expert {len(self.experts)}. Total Experts: {len(self.experts)}")

    def forward(self, x):
        B, T, C = x.size()
        if len(self.experts) == 0:
            return torch.zeros_like(x)
            
        x_flat = x.view(-1, C) # (N, C)
        
        # Distance-based routing: find nearest prototype
        # dist = ||x - p||^2 = ||x||^2 + ||p||^2 - 2xp^T
        # We can use cosine similarity for stability
        norms_x = F.normalize(x_flat, p=2, dim=-1)
        norms_p = F.normalize(self.prototypes, p=2, dim=-1)
        sim = norms_x @ norms_p.T # (N, num_experts)
        
        _, best_idx = torch.max(sim, dim=-1) # (N,)
        
        final_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (best_idx == i)
            if mask.any():
                final_output[mask] = expert(x_flat[mask])
                
        return final_output.view(B, T, C)

class PrototypeRocketTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, expert_hidden=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'ln1': nn.LayerNorm(d_model),
            'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            'ln2': nn.LayerNorm(d_model),
            'moe': PrototypeMoELayer(d_model, expert_hidden)
        }) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.emb(x) + self.pos
        for block in self.blocks:
            # Attention
            res = x
            attn_out, _ = block['attn'](block['ln1'](x), block['ln1'](x), block['ln1'](x), 
                                       need_weights=False, 
                                       attn_mask=torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=1).bool())
            x = res + attn_out
            
            # MoE
            res = x
            moe_out = block['moe'](block['ln2'](x))
            x = res + moe_out
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def train_prototype_rocket(model, x, y, task_id):
    # Grow if needed
    model.eval()
    with torch.no_grad():
        x_emb = model.emb(x[:100]) + model.pos
        # We use the mean embedding of the first layer's input as the prototype for simplicity
        # Actually, let's just use the mean of the tokens
        task_prototype = x_emb.mean(dim=(0, 1)) 
        
    print(f"Adding expert for Task {task_id} with prototype vector...")
    for block in model.blocks:
        block['moe'].add_expert(task_prototype)
        
    # Freeze old experts
    for block in model.blocks:
        for i, expert in enumerate(block['moe'].experts):
            if i < task_id:
                for p in expert.parameters(): p.requires_grad = False
            else:
                for p in expert.parameters(): p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(10):
        for i in range(0, len(x), BATCH_SIZE):
            bx, by = x[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]
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
    print(f"\n{'='*50}\nRunning Prototype-based MoE (Stable Structural Memory)\n{'='*50}")
    model = PrototypeRocketTransformer(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        train_prototype_rocket(model, x, y, t_id)
        
        for eval_id, (tx, ty) in enumerate(test_tasks):
            loss = evaluate(model, tx, ty)
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")

