import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

# --- 1. Synthetic Continual Learning Dataset ---
# We define 3 distinct tasks to test catastrophic forgetting.
# Task 0: Sequence of Even numbers
# Task 1: Sequence of Odd numbers
# Task 2: Sequence of Fibonacci numbers
# All framed as next-token prediction over a vocabulary of 100 tokens.

VOCAB_SIZE = 100
SEQ_LEN = 16
BATCH_SIZE = 64

def generate_task_data(task_id, num_samples):
    X, Y = [], []
    for _ in range(num_samples):
        if task_id == 0:
            # Even numbers 0, 2, 4, ...
            start = np.random.randint(0, 50) * 2
            seq = [(start + i * 2) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        elif task_id == 1:
            # Odd numbers 1, 3, 5, ...
            start = np.random.randint(0, 50) * 2 + 1
            seq = [(start + i * 2) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        else:
            # Random sequence, but with a specific offset rule to be distinct
            start = np.random.randint(0, VOCAB_SIZE)
            seq = [(start + i * 5) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        X.append(seq[:-1])
        Y.append(seq[1:])
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

class CLDataset:
    def __init__(self, task_id, num_samples):
        self.X, self.Y = generate_task_data(task_id, num_samples)
    def get_batches(self):
        for i in range(0, len(self.X), BATCH_SIZE):
            yield self.X[i:i+BATCH_SIZE], self.Y[i:i+BATCH_SIZE]

# --- 2. Baseline Transformer (The Thermodynamics Law) ---
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# --- 3. The "Rocket": Structural Expansion Transformer ---
# Uses standard T-blocks but dynamically adds parallel blocks 
# and routes based on structural memory to avoid forgetting.
class StructuralRocketTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        
        # We start with 1 path (expert), and can grow
        self.paths = nn.ModuleList([
            nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        ])
        
        # A simple router (memory index)
        self.router = nn.Linear(d_model, 1)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.current_task_path = 0

    def add_new_path(self):
        # Freeze existing paths
        for path in self.paths:
            for param in path.parameters():
                param.requires_grad = False
        
        # Add new path
        device = next(self.parameters()).device
        new_path = nn.ModuleList([TransformerBlock(self.d_model, self.n_heads) for _ in range(len(self.paths[0]))]).to(device)
        self.paths.append(new_path)
        
        # Expand router
        old_router_weight = self.router.weight.data
        old_router_bias = self.router.bias.data
        new_router = nn.Linear(self.d_model, len(self.paths))
        new_router.weight.data[:-1] = old_router_weight
        new_router.bias.data[:-1] = old_router_bias
        # Initialize new route path slightly higher so it gets selected for new data
        new_router.weight.data[-1] = torch.randn_like(new_router.weight.data[-1]) * 0.02
        new_router.bias.data[-1] = 0.1
        self.router = new_router.to(self.paths[0][0].ln_1.weight.device)
        self.current_task_path = len(self.paths) - 1
        
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Route at the sequence level based on the mean embedding
        route_logits = self.router(x.mean(dim=1)) # (B, num_paths)
        route_probs = F.softmax(route_logits, dim=-1) # (B, num_paths)
        
        # For simplicity in this early version, we do a soft mixture of paths
        out_x = torch.zeros_like(x)
        for path_idx, path in enumerate(self.paths):
            path_x = x
            for block in path:
                path_x = block(path_x)
            out_x = out_x + path_x * route_probs[:, path_idx].view(B, 1, 1)
            
        x = self.ln_f(out_x)
        return self.head(x)

# --- 4. Training and Evaluation Loop ---
def evaluate(model, dataset, device):
    model.eval()
    total_loss = 0
    batches = 0
    with torch.no_grad():
        for x, y in dataset.get_batches():
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            total_loss += loss.item()
            batches += 1
    return total_loss / batches if batches > 0 else 0

def train_task(model, dataset, optimizer, device, epochs=3):
    model.train()
    for ep in range(epochs):
        for x, y in dataset.get_batches():
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def run_experiment(model_type, device):
    print(f"\n{'='*50}\nRunning Experiment with {model_type}\n{'='*50}")
    
    if model_type == "Baseline":
        model = BaselineTransformer(VOCAB_SIZE, d_model=128, n_heads=4, n_layers=4).to(device)
    else:
        model = StructuralRocketTransformer(VOCAB_SIZE, d_model=128, n_heads=4, n_layers=4).to(device)
        
    print(f"Initial Trainable Params: {count_parameters(model):,}")
    
    tasks = [CLDataset(i, 2000) for i in range(3)]
    test_tasks = [CLDataset(i, 500) for i in range(3)]
    
    task_performances = []
    
    for task_id, task_data in enumerate(tasks):
        print(f"\n--- Training on Task {task_id} ---")
        
        if model_type == "Rocket" and task_id > 0:
            print("Expanding Rocket Structure...")
            model.add_new_path()
            print(f"Total Params (including frozen): {count_all_parameters(model):,}")
            print(f"Current Trainable Params: {count_parameters(model):,}")
            
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_task(model, task_data, optimizer, device, epochs=10)
        
        # Evaluate on all tasks seen so far (and future ones just to see)
        evals = []
        for eval_task_id, eval_data in enumerate(test_tasks):
            loss = evaluate(model, eval_data, device)
            evals.append(loss)
            status = "SEEN" if eval_task_id <= task_id else "UNSEEN"
            print(f"Eval Task {eval_task_id} ({status}) Loss: {loss:.4f}")
            
        task_performances.append(evals)
        
    return task_performances

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    res_base = run_experiment("Baseline", device)
    res_rocket = run_experiment("Rocket", device)
    
    print("\n\n" + "#"*50 + "\nRESULTS SUMMARY\n" + "#"*50)
    print("Baseline Catastrophic Forgetting (Task 0 Loss after training on Task 2):", res_base[2][0])
    print("Rocket Catastrophic Forgetting (Task 0 Loss after training on Task 2):", res_rocket[2][0])

