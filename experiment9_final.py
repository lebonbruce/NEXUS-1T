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

# --- The "Structural Memory" Layer (The Final Rocket) ---
class StructuralMemoryLayer(nn.Module):
    def __init__(self, d_model, top_k=4):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.register_buffer('keys', torch.empty(0, d_model))
        self.register_buffer('values', torch.empty(0, d_model))
        # Lightweight router to map hidden state to memory query
        self.query_proj = nn.Linear(d_model, d_model)
        
    def add_memory(self, k, v):
        # k: (N, d_model), v: (N, d_model)
        self.keys = torch.cat([self.keys, k], dim=0)
        self.values = torch.cat([self.values, v], dim=0)
        print(f"[Memory] Growing to {self.keys.size(0)} entries.")

    def forward(self, x):
        B, T, C = x.size()
        if self.keys.size(0) == 0:
            return torch.zeros_like(x)
            
        q = self.query_proj(x) # (B, T, d_model)
        
        # Sparse Attention over memory
        # sim = (q @ keys^T)
        sim = torch.matmul(q, self.keys.T) # (B, T, num_mem)
        
        # Pick top-k memories
        k = min(self.top_k, self.keys.size(0))
        top_sim, top_idx = torch.topk(sim, k=k, dim=-1)
        
        # Gather top-k values
        # top_idx: (B, T, k)
        # self.values: (num_mem, d_model)
        flat_idx = top_idx.view(-1)
        selected_values = self.values[flat_idx].view(B, T, k, C)
        
        # Weighted sum
        weights = F.softmax(top_sim, dim=-1).unsqueeze(-1) # (B, T, k, 1)
        out = (weights * selected_values).sum(dim=2) # (B, T, d_model)
        
        return out

class FinalRocketTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'ln1': nn.LayerNorm(d_model),
            'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            'ln2': nn.LayerNorm(d_model),
            'mem': StructuralMemoryLayer(d_model)
        }) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.emb(x) + self.pos
        for block in self.blocks:
            res = x
            attn_out, _ = block['attn'](block['ln1'](x), block['ln1'](x), block['ln1'](x), 
                                       need_weights=False, 
                                       attn_mask=torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=1).bool())
            x = res + attn_out
            
            res = x
            x = res + block['mem'](block['ln2'](x))
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def train_final_rocket(model, x, y, task_id):
    # Determine what knowledge is missing (surprise)
    model.eval()
    with torch.no_grad():
        logits = model(x[:200])
        pred = torch.argmax(logits, dim=-1)
        # Correct tokens give us the "Good" representations to store as memory
        # Incorrect tokens give us the "Surprising" ones
        correct_mask = (pred == y[:200])
        # We store the representations of tokens that were correctly predicted after training?
        # Actually, let's just store the representations of the current task.
        
    print(f"Storing Task {task_id} knowledge into structural memory...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for ep in range(10):
        for i in range(0, len(x), BATCH_SIZE):
            bx, by = x[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), by.view(-1))
            loss.backward()
            optimizer.step()
            
    # After training, "Inject" the learned representations into frozen structural memory
    model.eval()
    with torch.no_grad():
        # Capture representations from the first block's output
        # (Using a small subset to avoid memory explosion)
        bx = x[:100]
        x_in = model.emb(bx) + model.pos
        for block in model.blocks:
            k = block['ln2'](x_in).view(-1, model.emb.embedding_dim) # Keys are the representations
            v = k # For simplicity, Values are also the representations (Auto-associative)
            # In a real model, Value would be the target update or the next-token prediction
            block['mem'].add_memory(k[:50], v[:50]) # Inject 50 representative neurons

if __name__ == "__main__":
    print(f"\n{'='*50}\nRunning FinalRocket (Infinite KV Memory Transformer)\n{'='*50}")
    model = FinalRocketTransformer(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        train_final_rocket(model, x, y, t_id)
        
        for eval_id, (tx, ty) in enumerate(test_tasks):
            model.eval()
            with torch.no_grad():
                logits = model(tx)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), ty.view(-1)).item()
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")

