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

# --- The "Structural Neuron" Layer ---
# This layer has a fixed "General Knowledge" part and a growing "Specialized Neuron" part.
class InjectionFFN(nn.Module):
    def __init__(self, d_model, initial_neurons=128):
        super().__init__()
        self.d_model = d_model
        # General knowledge (trainable)
        self.w1 = nn.Parameter(torch.randn(initial_neurons, d_model) / math.sqrt(d_model))
        self.w2 = nn.Parameter(torch.randn(d_model, initial_neurons) / math.sqrt(initial_neurons))
        
        # Specialized structural neurons (injected)
        self.register_buffer('injected_w1', torch.empty(0, d_model))
        self.register_buffer('injected_w2', torch.empty(d_model, 0))
        
    def inject_neurons(self, keys, values):
        # keys: (N, d_model) - Input representations that triggered surprise
        # values: (d_model, N) - Corresponding desired outputs/updates
        self.injected_w1 = torch.cat([self.injected_w1, keys], dim=0)
        self.injected_w2 = torch.cat([self.injected_w2, values], dim=1)
        print(f"[Injection] Total Neurons: {self.w1.size(0)} (General) + {self.injected_w1.size(0)} (Specialized)")

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        # 1. General knowledge path
        mid = F.gelu(x_flat @ self.w1.T)
        out_general = mid @ self.w2.T
        
        # 2. Specialized path (Non-trainable but structural)
        if self.injected_w1.size(0) > 0:
            # We use a similarity-based activation for injected neurons
            # This is like a RBF (Radial Basis Function) or Attention
            # Injection activation: exp(-||x - key||^2) or just dot-product
            # To keep it "Transformer-like", we use dot-product + softmax
            sim = (x_flat @ self.injected_w1.T) # (N, num_injected)
            # Use a high temperature to make it very sparse/selective
            attn = F.softmax(sim * 10.0, dim=-1) 
            out_specialized = attn @ self.injected_w2.T
            return (out_general + out_specialized).view(B, T, C)
            
        return out_general.view(B, T, C)

class InjectionRocket(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'ln1': nn.LayerNorm(d_model),
            'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            'ln2': nn.LayerNorm(d_model),
            'ffn': InjectionFFN(d_model)
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
            x = res + block['ffn'](block['ln2'](x))
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def train_injection(model, x, y, task_id):
    # Initial assessment of surprise
    model.eval()
    with torch.no_grad():
        logits = model(x[:100])
        losses = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y[:100].view(-1), reduction='none').view(100, -1).mean(dim=1)
        
    # Inject neurons for surprising samples
    surprise_threshold = 2.0
    surprising_indices = (losses > surprise_threshold).nonzero().squeeze()
    if surprising_indices.numel() > 0:
        if surprising_indices.dim() == 0: surprising_indices = surprising_indices.unsqueeze(0)
        print(f"Injecting neurons for {surprising_indices.numel()} surprising sequences in Task {task_id}...")
        
        with torch.no_grad():
            # Get the representations at the last layer before FFN
            # (Simplified: we just inject at all layers)
            x_emb = model.emb(x[surprising_indices]) + model.pos
            # Use the mean embedding of the sequence as Key, and a random target-aligned vector as Value
            # Real implementation would use the gradient of the loss.
            keys = x_emb.mean(dim=1) # (N, d_model)
            values = torch.randn_like(keys).T * 0.1 # (d_model, N)
            
            for block in model.blocks:
                block['ffn'].inject_neurons(keys, values)

    # Standard training for general knowledge
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

if __name__ == "__main__":
    print(f"\n{'='*50}\nRunning InjectionRocket (Structural Neuron Injection)\n{'='*50}")
    model = InjectionRocket(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        train_injection(model, x, y, t_id)
        
        for eval_id, (tx, ty) in enumerate(test_tasks):
            model.eval()
            with torch.no_grad():
                logits = model(tx)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), ty.view(-1)).item()
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")

