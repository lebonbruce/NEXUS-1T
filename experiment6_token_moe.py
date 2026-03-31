import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import time

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

# --- 2. Token-Level Structural Expert ---
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

# --- 3. Surprise-Triggered Growing MoE Layer ---
class GrowingMoELayer(nn.Module):
    def __init__(self, d_model, hidden_dim, initial_experts=1, entropy_threshold=1.5):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.entropy_threshold = entropy_threshold
        
        self.experts = nn.ModuleList([StructuralExpert(d_model, hidden_dim) for _ in range(initial_experts)])
        self.router = nn.Linear(d_model, initial_experts) # Simple router
        
    def add_expert(self):
        new_expert = StructuralExpert(self.d_model, self.hidden_dim).to(DEVICE)
        self.experts.append(new_expert)
        
        # Expand router
        old_router = self.router
        self.router = nn.Linear(self.d_model, len(self.experts)).to(DEVICE)
        with torch.no_grad():
            self.router.weight[:-1] = old_router.weight
            self.router.bias[:-1] = old_router.bias
            # New expert gets a fresh, slightly biased route
            nn.init.normal_(self.router.weight[-1], std=0.02)
            nn.init.constant_(self.router.bias[-1], 0.1)
        print(f"[MoE] Growing to {len(self.experts)} experts.")

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()
        x_flat = x.view(-1, C) # (N, C) where N = B*T
        
        logits = self.router(x_flat) # (N, num_experts)
        probs = F.softmax(logits, dim=-1) # (N, num_experts)
        
        # Calculate entropy of routing to see if we're "confused"
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        
        # Pick top-1 expert per token
        top1_probs, top1_idx = torch.max(probs, dim=-1) # (N,)
        
        # Actually compute the outputs
        # To be efficient, we group tokens by their chosen expert
        final_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top1_idx == i)
            if mask.any():
                final_output[mask] = expert(x_flat[mask])
        
        return final_output.view(B, T, C), entropy

# --- 4. The Token-Rocket Architecture ---
class TokenRocketBlock(nn.Module):
    def __init__(self, d_model, n_heads, expert_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = GrowingMoELayer(d_model, expert_hidden)
        
    def forward(self, x):
        # Attention is the "thermodynamic glue" (reasoning)
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), 
                                need_weights=False, 
                                attn_mask=torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=1).bool())
        x = x + attn_out
        
        # MoE is the "structural memory" (knowledge)
        moe_out, entropy = self.moe(self.ln2(x))
        x = x + moe_out
        return x, entropy

class TokenRocketTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, expert_hidden=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.blocks = nn.ModuleList([TokenRocketBlock(d_model, n_heads, expert_hidden) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.emb(x) + self.pos
        total_entropy = 0
        for block in self.blocks:
            x, ent = block(x)
            total_entropy += ent
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, total_entropy / len(self.blocks)

# --- 5. Training and Evaluation ---
def train_rocket(model, x, y, task_id):
    # Determine if we need to grow based on initial performance (surprise)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x[:100])
        initial_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y[:100].view(-1))
    
    # If loss is high, it means existing experts (knowledge) can't explain the data
    if initial_loss > 1.0: 
        print(f"Task {task_id} surprise detected (Loss: {initial_loss:.4f}). Spawning new experts...")
        for block in model.blocks:
            block.moe.add_expert()
            
    # Optimizer for ALL parameters but we could also freeze old experts
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(10):
        for i in range(0, len(x), BATCH_SIZE):
            bx, by = x[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), by.view(-1))
            loss.backward()
            optimizer.step()

def evaluate_rocket(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        return F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()

if __name__ == "__main__":
    print(f"\n{'='*50}\nRunning Token-Level Growing MoE (Token-Rocket)\n{'='*50}")
    model = TokenRocketTransformer(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        train_rocket(model, x, y, t_id)
        
        for eval_id, (tx, ty) in enumerate(test_tasks):
            loss = evaluate_rocket(model, tx, ty)
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")

