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

# --- 2. The Delta-LoRA Hybrid Layer ---
# Combines a fast-weight state (Delta Rule) with task-specific LoRA adapters.
class DeltaLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # 1. Base Weights (Frozen after Task 0 or pre-trained)
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad = False
        
        # 2. Structural Memory (LoRA Adapters)
        self.adapters_a = nn.ParameterList([])
        self.adapters_b = nn.ParameterList([])
        
        # 3. Dynamic Memory (Delta State)
        self.head_dim = 16 # Sub-dimension for fast associative memory
        self.n_heads = out_features // self.head_dim
        self.register_buffer('delta_state', torch.zeros(1, self.n_heads, self.head_dim, self.head_dim))

    def add_adapter(self):
        a = nn.Parameter(torch.randn(self.in_features, self.rank, device=DEVICE) / math.sqrt(self.in_features))
        b = nn.Parameter(torch.zeros(self.rank, self.out_features, device=DEVICE))
        self.adapters_a.append(a)
        self.adapters_b.append(b)

    def forward(self, x, adapter_idx=None, delta_state=None):
        # x: (B, T, in)
        B, T, _ = x.size()
        
        # Path 1: Base + Adapters
        out = self.base(x)
        if adapter_idx is not None and adapter_idx < len(self.adapters_a):
            out = out + (x @ self.adapters_a[adapter_idx]) @ self.adapters_b[adapter_idx]
            
        # Path 2: Delta Fast Weights (Short-term context)
        # We simplify the Delta path here to be a residual state-based associative memory
        if delta_state is not None:
            # Associative retrieval: out = delta_state @ query(x)
            # This allows the model to "recall" facts from the current context state
            q = x.view(B, T, self.n_heads, self.head_dim)
            delta_out = torch.matmul(delta_state.unsqueeze(1), q.unsqueeze(-1)).squeeze(-1)
            out = out + delta_out.view(B, T, -1)
            
        return out

# --- 3. Streaming Evolution Rocket ---
class StreamingRocketBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = DeltaLoRALinear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = DeltaLoRALinear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.n_heads = n_heads

    def add_adapter(self):
        self.q_proj.add_adapter()
        self.v_proj.add_adapter()

    def forward(self, x, adapter_idx=None, delta_states=None):
        B, T, C = x.size()
        res = x
        x = self.ln1(x)
        
        # Delta states: list of (B, H, D, D)
        q = self.q_proj(x, adapter_idx, delta_states[0] if delta_states else None)
        k = self.k_proj(x)
        v = self.v_proj(x, adapter_idx, delta_states[1] if delta_states else None)
        
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = res + attn
        
        res = x
        x = self.ln2(x)
        x = res + self.mlp(x)
        return x

class StreamingRocket(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.blocks = nn.ModuleList([StreamingRocketBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.num_adapters = 0

    def add_adapter(self):
        for b in self.blocks: b.add_adapter()
        self.num_adapters += 1

    def forward(self, x, adapter_idx=None, delta_states=None):
        x = self.emb(x) + self.pos
        # delta_states: list of lists (per block)
        for i, block in enumerate(self.blocks):
            x = block(x, adapter_idx, delta_states[i] if delta_states else None)
        x = self.ln_f(x)
        return self.head(x)

# --- 4. Training and Evolution Loop ---
def train_streaming(model, x, y, task_id):
    # 1. Assessment (Surprise Detection)
    model.eval()
    with torch.no_grad():
        logits = model(x[:100], adapter_idx=task_id-1 if task_id > 0 else None)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y[:100].view(-1))
    
    # 2. Structural Growth Trigger
    if loss > 1.0 or task_id == 0:
        model.add_adapter()
        print(f"[Rocket] Surprise detected ({loss:.4f}). Structural expansion to {model.num_adapters} stages.")

    # 3. Evolution: Only train the new structural stage
    active_idx = model.num_adapters - 1
    params = [p for n, p in model.named_parameters() if f"adapters_a.{active_idx}" in n or f"adapters_b.{active_idx}" in n]
    if task_id == 0:
        # Initial task trains the whole skeleton
        params = model.parameters()
        
    optimizer = optim.Adam(params, lr=1e-3)
    model.train()
    for ep in range(10):
        for i in range(0, len(x), BATCH_SIZE):
            bx, by = x[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(bx, adapter_idx=active_idx)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), by.view(-1))
            loss.backward()
            optimizer.step()

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        # Routing: try all structural stages and pick the best (Automatic Routing)
        best_loss = 1e9
        for i in range(model.num_adapters):
            logits = model(x, adapter_idx=i)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()
            best_loss = min(best_loss, loss)
        return best_loss

if __name__ == "__main__":
    print(f"\n{'='*50}\nRunning Streaming Rocket (Hybrid State + Structure)\n{'='*50}")
    model = StreamingRocket(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        train_streaming(model, x, y, t_id)
        
        for eval_id, (tx, ty) in enumerate(test_tasks):
            loss = evaluate(model, tx, ty)
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")

