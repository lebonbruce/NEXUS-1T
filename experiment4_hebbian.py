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
        X.append(seq[:-1])
        Y.append(seq[1:])
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

class CLDataset:
    def __init__(self, task_id, num_samples):
        self.X, self.Y = generate_task_data(task_id, num_samples)
    def get_batches(self):
        for i in range(0, len(self.X), BATCH_SIZE):
            yield self.X[i:i+BATCH_SIZE], self.Y[i:i+BATCH_SIZE]

# --- 2. The Hebbian Fast-Weight Layer ---
# This layer maintains a "slow weight" (trained by backprop)
# and a "fast weight" (updated during forward pass via Hebbian rule).
class HebbianLinear(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.1, eta=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.slow_weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.alpha = alpha # Decay rate of fast weights
        self.eta = eta     # Learning rate of fast weights
        self.register_buffer('fast_weight', torch.zeros(out_features, in_features))

    def forward(self, x):
        # x: (B, T, in_features)
        B, T, C = x.size()
        
        # Associative Memory approximation:
        # Instead of sequential updates, we compute the fast weight as a 
        # discounted sum of the correlations in the current sequence.
        # W_fast_t = eta * sum_{i=0}^{t-1} (alpha^(t-1-i) * y_i * x_i^T)
        # For simplicity and speed in this prototype, we use a global sequence-level 
        # fast weight generated from the sequence itself (Self-Referential Fast Weights).
        
        # 1. Slow path
        slow_out = F.linear(x, self.slow_weight) # (B, T, out)
        
        # 2. Fast path (Simplified: Fast weights are a function of the input sequence)
        # This acts like a linear attention mechanism or a kernel memory.
        # We use the previous tokens to "program" the weights for the current token.
        # W_fast = eta * (x^T @ x) -> but we need it per-sequence.
        # Let's use the Linear Attention identity: (Query @ (Key^T @ Value))
        # Here, y = (W_slow + W_fast) @ x = W_slow@x + (eta * sum(y_prev @ x_prev^T)) @ x
        
        # For this experiment, let's use the "Delta Rule" or "Linear Attention" equivalent:
        # We'll use a simplified version: fast weights are transient.
        # Since we want "learning while inferring", we'll use the cumulative correlation.
        
        # Implementation of Linear Attention as a Fast Weight:
        # y_t = W_slow @ x_t + sum_{i<t} (x_i @ x_i^T) @ x_t
        # This is equivalent to standard linear attention.
        
        # To make it truly "Hebbian", we'll stick to the sequential logic but use a faster implementation
        # if possible. For now, let's just keep it simple but ensure it's on the right device.
        
        outputs = []
        current_fast_weight = self.fast_weight.clone().unsqueeze(0).expand(B, -1, -1) # (B, out, in)
        
        for t in range(T):
            xt = x[:, t, :].unsqueeze(2) # (B, C, 1)
            yt = (self.slow_weight.unsqueeze(0) @ xt) + (current_fast_weight @ xt) # (B, out, 1)
            
            # Local update
            current_fast_weight = self.alpha * current_fast_weight + self.eta * (yt @ xt.transpose(1, 2))
            outputs.append(yt.squeeze(2))
            
        return torch.stack(outputs, dim=1)

# --- 3. The Hebbian-T Architecture ---
class HebbianAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # Query is standard, Key/Value use Hebbian plastic layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = HebbianLinear(d_model, d_model)
        self.v_proj = HebbianLinear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class HebbianBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = HebbianAttention(d_model, n_heads)
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

class HebbianTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.ModuleList([HebbianBlock(d_model, n_heads) for _ in range(n_layers)])
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

# --- 4. Evaluation and Training ---
def evaluate(model, dataset, device):
    model.eval()
    total_loss = 0; batches = 0
    with torch.no_grad():
        for x, y in dataset.get_batches():
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            total_loss += loss.item()
            batches += 1
    return total_loss / batches if batches > 0 else 0

def train(model, dataset, optimizer, device, epochs=10):
    model.train()
    for ep in range(epochs):
        for x, y in dataset.get_batches():
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tasks = [CLDataset(i, 2000) for i in range(3)]
    test_tasks = [CLDataset(i, 500) for i in range(3)]
    
    print(f"\n{'='*50}\nRunning Hebbian Fast-Weight Transformer\n{'='*50}")
    model = HebbianTransformer(VOCAB_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for task_id, task_data in enumerate(tasks):
        print(f"\n--- Training on Task {task_id} ---")
        train(model, task_data, optimizer, device, epochs=10)
        
        for eval_task_id, eval_data in enumerate(test_tasks):
            loss = evaluate(model, eval_data, device)
            print(f"Eval Task {eval_task_id} Loss: {loss:.4f}")

