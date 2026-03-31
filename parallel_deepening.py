import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import time

# =============================================================================
# Parallel Deepening: Associative Memory + Surprise-Driven Branching
# =============================================================================

VOCAB_SIZE = 1000
SEQ_LEN = 32
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KnowledgeStream:
    def __init__(self, vocab_size=1000, seq_len=32):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
    def next_batch(self, batch_size, task_id):
        X = []
        for _ in range(batch_size):
            if task_id % 3 == 0: # Task type 0
                start = np.random.randint(0, self.vocab_size)
                step = np.random.randint(1, 10)
                seq = [(start + i * step) % self.vocab_size for i in range(self.seq_len + 1)]
            elif task_id % 3 == 1: # Task type 1
                half = self.seq_len // 2
                first_half = np.random.randint(0, self.vocab_size, size=half).tolist()
                seq = first_half + first_half[::-1] + [np.random.randint(0, 10)]
            else: # Task type 2
                start = np.random.randint(0, self.vocab_size)
                seq = [(start + i * i) % self.vocab_size for i in range(self.seq_len + 1)]
            X.append(seq)
        data = torch.tensor(X, dtype=torch.long, device=DEVICE)
        return data[:, :-1], data[:, 1:]

# --- Parallel Associative Memory (PAM) ---
# Instead of sequential Delta-rule, we use Parallel Associative Updates
class ParallelAssociativeMemory(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.size()
        H, D = self.n_heads, self.head_dim
        
        q = self.q_proj(x).view(B, T, H, D)
        k = F.normalize(self.k_proj(x).view(B, T, H, D), p=2, dim=-1)
        v = self.v_proj(x).view(B, T, H, D)
        
        # Parallel associative update:
        # Instead of sequential W_t, we use a causal cumulative sum of KV pairs
        # W_t = sum_{i < t} v_i @ k_i^T
        # Result_t = W_t @ q_t = sum_{i < t} (v_i @ k_i^T) @ q_t = sum_{i < t} v_i @ (k_i^T @ q_t)
        # This is exactly Linear Attention!
        
        # Linear Attention implementation:
        # (B, H, T, D) for q, k, v
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Causal linear attention:
        # Numerator: cumsum(v @ k^T) @ q
        # We use the kernel trick: Result = causal_cumsum(k) * something? No.
        # Standard causal linear attention is:
        # z_t = sum_{i=1}^t (v_i * k_i^T) q_t
        
        # Optimized implementation using torch.einsum or matmul
        # For simplicity and correctness, let's use the iterative form but vectorized over batch/heads
        # OR use the full matrix form for short sequences:
        # Masked matmul: (q @ k.T) * mask @ v
        # Wait, that's standard attention. Linear attention is (q @ (k.T @ v))
        
        # Correct Linear Causal Attention:
        # s_t = s_{t-1} + v_t @ k_t^T
        # y_t = s_t @ q_t
        
        # We can use a scan/cumsum for this
        # kv = v.unsqueeze(-1) @ k.unsqueeze(-2) # (B, H, T, D, D)
        # s = torch.cumsum(kv, dim=2) # (B, H, T, D, D)
        # out = torch.matmul(s, q.unsqueeze(-1)).squeeze(-1) # (B, H, T, D)
        
        # To avoid D*D memory, we use the property: (v_i @ k_i^T) @ q_t = v_i @ (k_i^T @ q_t)
        # But for causal, we need the sum for EACH t.
        # This can be done via (B, H, T, T) weight matrix where W_ij = (k_j @ q_i) if j <= i
        
        weights = torch.matmul(q, k.transpose(-2, -1)) # (B, H, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        weights = weights * mask
        
        out = torch.matmul(weights, v) # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)

# --- Surprise-Driven Organic FFN ---
class SurpriseFFN(nn.Module):
    def __init__(self, d_model, hidden_dim=512):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )])
        self.active_idx = 0
        
    def branch(self):
        # Clone the last expert to start from a good point, or start fresh
        new_expert = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.d_model)
        ).to(DEVICE)
        # Freeze previous
        for p in self.experts[self.active_idx].parameters():
            p.requires_grad = False
            
        self.experts.append(new_expert)
        self.active_idx = len(self.experts) - 1
        print(f"  [Surprise] Branching to expert {self.active_idx}")

    def forward(self, x):
        return self.experts[self.active_idx](x)

class ParallelRocket(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': ParallelAssociativeMemory(d_model, 4),
                'ln2': nn.LayerNorm(d_model),
                'ffn': SurpriseFFN(d_model)
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.emb(x) + self.pos
        for l in self.layers:
            x = x + l['attn'](l['ln1'](x))
            x = x + l['ffn'](l['ln2'](x))
        return self.head(self.ln_f(x))

def run_parallel_deepening():
    print(f"🚀 Initializing Parallel Deepening Engine...")
    stream = KnowledgeStream()
    model = ParallelRocket(VOCAB_SIZE).to(DEVICE)
    
    for task_id in range(5):
        print(f"\n--- Task Era {task_id} ---")
        
        # Initial surprise check
        model.eval()
        with torch.no_grad():
            bx, by = stream.next_batch(BATCH_SIZE, task_id)
            logits = model(bx)
            initial_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1))
            print(f"  Initial Surprise (Loss): {initial_loss.item():.4f}")
            
            if initial_loss > 2.0 and task_id > 0:
                for l in model.layers:
                    l['ffn'].branch()
        
        # Train
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for step in range(300):
            bx, by = stream.next_batch(BATCH_SIZE, task_id)
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1))
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"  Step {step:3d} | Loss: {loss.item():.4f}")
        
        # Eval history
        print(f"  [Eval] Task Retentions:")
        for old_id in range(task_id + 1):
            ex, ey = stream.next_batch(100, old_id)
            with torch.no_grad():
                l = F.cross_entropy(model(ex).reshape(-1, VOCAB_SIZE), ey.reshape(-1)).item()
                print(f"    Task {old_id}: {l:.4f}")

if __name__ == "__main__":
    run_parallel_deepening()
