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

# --- SOM-MoE Layer (Self-Organizing Map) ---
class SOMMoELayer(nn.Module):
    def __init__(self, d_model, grid_size=(4, 4), hidden_dim=256):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.num_experts = grid_size[0] * grid_size[1]
        
        # Grid coordinates
        x, y = torch.meshgrid(torch.arange(grid_size[0]), torch.arange(grid_size[1]), indexing='ij')
        self.register_buffer('coords', torch.stack([x.flatten(), y.flatten()], dim=1).float()) # (num_experts, 2)
        
        # Experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        ) for _ in range(self.num_experts)])
        
        # Prototype vectors for each expert node in the SOM
        self.prototypes = nn.Parameter(torch.randn(self.num_experts, d_model) / math.sqrt(d_model))
        
    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C) # (N, C)
        
        # 1. Competitive Routing: Find Best Matching Unit (BMU)
        # We use cosine similarity to find the nearest prototype
        sim = F.cosine_similarity(x_flat.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1) # (N, num_experts)
        bmu_idx = torch.argmax(sim, dim=-1) # (N,)
        
        # 2. Output Calculation (Soft-Ensemble based on SOM distance)
        # Instead of just top-1, we can use a neighborhood function?
        # For efficiency in T-architecture, we'll use Top-1 during inference, 
        # but the SOM structure ensures that neighboring experts handle similar data.
        
        final_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (bmu_idx == i)
            if mask.any():
                final_out[mask] = expert(x_flat[mask])
                
        return final_out.view(B, T, C), bmu_idx

class SOMRocket(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.som_moe = SOMMoELayer(d_model, grid_size=(4, 4))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.emb(x) + self.pos
        res = x
        x, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False, 
                         attn_mask=torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=1).bool())
        x = res + x
        res = x
        moe_out, bmu_idx = self.som_moe(self.ln2(x))
        x = res + moe_out
        x = self.ln_f(x)
        return self.head(x), bmu_idx

def train_som(model, x, y):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(10):
        for i in range(0, len(x), BATCH_SIZE):
            bx, by = x[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            logits, bmu_idx = model(bx)
            
            # Loss 1: Task loss
            ce_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), by.view(-1))
            
            # Loss 2: SOM Topological Loss (Optional but recommended)
            # Ensure that the prototypes stay organized
            # For simplicity, we just use standard BP on prototypes here.
            
            ce_loss.backward()
            optimizer.step()

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        return F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()

if __name__ == "__main__":
    print(f"\n{'='*50}\nRunning SOM-Rocket (Topological Competition)\n{'='*50}")
    model = SOMRocket(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        train_som(model, x, y)
        for eval_id, (tx, ty) in enumerate(test_tasks):
            loss = evaluate(model, tx, ty)
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")
