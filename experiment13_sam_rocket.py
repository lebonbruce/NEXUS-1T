import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

# --- 1. Synthetic Continual Learning Dataset ---
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

# --- 2. Self-Assembling Matrix (SAM) Layer ---
# This layer uses a giant memory bank. Inputs route to specific slots.
# Only the "addressed" slots are used/updated.
class SAMLayer(nn.Module):
    def __init__(self, d_model, memory_slots=4096, top_k=1):
        super().__init__()
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.top_k = top_k
        
        # Physical Memory Matrix
        self.memory_w1 = nn.Parameter(torch.randn(memory_slots, d_model) / math.sqrt(d_model))
        self.memory_w2 = nn.Parameter(torch.randn(memory_slots, d_model) / math.sqrt(d_model))
        
        # Router: maps input to memory addresses
        self.router = nn.Linear(d_model, memory_slots)
        
    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C) # (N, C)
        
        # 1. Routing
        logits = self.router(x_flat) # (N, slots)
        
        # Competitive selection: Top-K slots
        # To prevent collapse, we could add noise or entropy regularization
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Softmax over top-k for gating
        weights = F.softmax(topk_logits, dim=-1) # (N, top_k)
        
        # 2. Computation (Sparse)
        # We gather the weights for the selected slots
        # memory_w1: (slots, C) -> selected_w1: (N, top_k, C)
        # memory_w2: (slots, C) -> selected_w2: (N, top_k, C)
        
        # Efficient gathering
        # Instead of giant matmuls, we do element-wise product since it's top-k=1 usually
        if self.top_k == 1:
            idx = topk_indices.squeeze(-1) # (N,)
            w1 = self.memory_w1[idx] # (N, C)
            w2 = self.memory_w2[idx] # (N, C)
            
            # Simulated FFN: y = activation(x * w1) * w2
            # For simplicity, we'll use dot-product as a similarity feature
            mid = torch.sum(x_flat * w1, dim=-1, keepdim=True) # (N, 1)
            mid = F.gelu(mid)
            out = mid * w2 # (N, C)
        else:
            # Multi-slot support
            out = torch.zeros_like(x_flat)
            for k in range(self.top_k):
                idx = topk_indices[:, k]
                w1 = self.memory_w1[idx]
                w2 = self.memory_w2[idx]
                mid = torch.sum(x_flat * w1, dim=-1, keepdim=True)
                mid = F.gelu(mid)
                out += weights[:, k:k+1] * (mid * w2)
                
        return out.view(B, T, C)

# --- 3. SAM-Rocket Transformer ---
class SAMRocket(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ln2': nn.LayerNorm(d_model),
                'sam': SAMLayer(d_model, memory_slots=8192, top_k=1)
            }) for _ in range(n_layers)
        ])
        
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
            
            # SAM (Memory)
            res = x
            x = res + block['sam'](block['ln2'](x))
            
        x = self.ln_f(x)
        return self.head(x)

# --- 4. Training Loop ---
def train_sam(model, x, y, task_id):
    # To ensure experts don't collapse, we add a routing loss (load balancing)
    # But for continual learning, we actually WANT experts to specialize.
    # We will use a "Freezing" mechanism or a very low LR for old experts.
    
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
    print(f"\n{'='*50}\nRunning SAM-Rocket (Self-Assembling Matrix)\n{'='*50}")
    model = SAMRocket(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        train_sam(model, x, y, t_id)
        
        for eval_id, (tx, ty) in enumerate(test_tasks):
            loss = evaluate(model, tx, ty)
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")

