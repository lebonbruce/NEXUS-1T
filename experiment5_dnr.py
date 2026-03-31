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

# --- 2. LoRA-style Dynamic Adapter ---
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=4):
        super().__init__()
        self.base_layer = base_layer
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        self.rank = rank
        self.adapters_a = nn.ParameterList([])
        self.adapters_b = nn.ParameterList([])
        
    def add_adapter(self):
        in_dim = self.base_layer.in_features
        out_dim = self.base_layer.out_features
        device = next(self.base_layer.parameters()).device
        a = nn.Parameter(torch.randn(in_dim, self.rank, device=device) / math.sqrt(in_dim))
        b = nn.Parameter(torch.zeros(self.rank, out_dim, device=device)) # Initialize B as zero to start with 0 impact
        self.adapters_a.append(a)
        self.adapters_b.append(b)

    def forward(self, x, adapter_idx=None):
        # x: (B, T, in_dim)
        base_out = self.base_layer(x)
        if adapter_idx is None or len(self.adapters_a) == 0:
            return base_out
            
        # Add adapter contribution: x @ A @ B
        # adapter_idx can be a single index or a tensor of indices (B,)
        if isinstance(adapter_idx, int):
            a = self.adapters_a[adapter_idx]
            b = self.adapters_b[adapter_idx]
            adapter_out = (x @ a) @ b
            return base_out + adapter_out
        else:
            # Batched adapter selection
            # For simplicity in this prototype, we'll assume a single active adapter for training
            a = self.adapters_a[adapter_idx[0]]
            b = self.adapters_b[adapter_idx[0]]
            return base_out + (x @ a) @ b

# --- 3. DNR Rocket Architecture ---
class DNRTBlock(nn.Module):
    def __init__(self, d_model, n_heads, rank=8):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        # Standard attention, but we wrap the projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.q_lora = LoRALinear(self.q_proj, rank)
        self.v_lora = LoRALinear(self.v_proj, rank)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp_in = nn.Linear(d_model, 4 * d_model)
        self.mlp_out = nn.Linear(4 * d_model, d_model)
        self.mlp_lora = LoRALinear(self.mlp_out, rank)
        self.n_heads = n_heads

    def add_adapter(self):
        self.q_lora.add_adapter()
        self.v_lora.add_adapter()
        self.mlp_lora.add_adapter()

    def forward(self, x, adapter_idx=None):
        B, T, C = x.size()
        res = x
        x = self.ln_1(x)
        
        q = self.q_lora(x, adapter_idx).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.v_lora(x, adapter_idx).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = res + self.o_proj(attn)
        
        res = x
        x = self.ln_2(x)
        x = self.mlp_in(x)
        x = F.gelu(x)
        x = res + self.mlp_lora(x, adapter_idx)
        return x

class DNRTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.ModuleList([DNRTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.num_adapters = 0
        self.entropy_threshold = 2.0

    def add_adapter(self):
        for block in self.blocks:
            block.add_adapter()
        self.num_adapters += 1
        print(f"[DNR] Added Adapter {self.num_adapters}")

    def forward(self, idx, adapter_idx=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # If no adapter specified, we try to route
        if adapter_idx is None:
            if self.num_adapters == 0:
                pass # Use base model
            else:
                # Calculate entropy with each adapter to choose the best one
                # For this prototype, we just return the list of entropies
                # and let the trainer decide, or use the last one for training.
                pass
                
        for block in self.blocks:
            x = block(x, adapter_idx)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Calculate entropy for routing
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        
        return logits, entropy

# --- 4. Training Loop with Dynamic Adapter Spawning ---
def train_dnr(model, dataset, device, epochs=10):
    model.train()
    
    # Check if we need a new adapter
    x_first, _ = next(dataset.get_batches())
    x_first = x_first.to(device)
    with torch.no_grad():
        _, entropy = model(x_first)
    
    if entropy > model.entropy_threshold or model.num_adapters == 0:
        model.add_adapter()
        
    active_idx = model.num_adapters - 1
    # Only train the newly added adapter parameters
    trainable_params = []
    for name, param in model.named_parameters():
        if f"adapters_a.{active_idx}" in name or f"adapters_b.{active_idx}" in name:
            param.requires_grad = True
            trainable_params.append(param)
        elif "adapters" in name:
            param.requires_grad = False
            
    optimizer = optim.Adam(trainable_params, lr=1e-3)
    
    for ep in range(epochs):
        for x, y in dataset.get_batches():
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x, adapter_idx=active_idx)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()

def evaluate_dnr(model, dataset, device):
    model.eval()
    # Try all adapters and pick the best one per batch (routing)
    total_loss = 0; batches = 0
    with torch.no_grad():
        for x, y in dataset.get_batches():
            x, y = x.to(device), y.to(device)
            
            best_batch_loss = float('inf')
            for i in range(model.num_adapters):
                logits, _ = model(x, adapter_idx=i)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                if loss < best_batch_loss:
                    best_batch_loss = loss.item()
            
            total_loss += best_batch_loss
            batches += 1
    return total_loss / batches if batches > 0 else 0

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tasks = [CLDataset(i, 2000) for i in range(3)]
    test_tasks = [CLDataset(i, 500) for i in range(3)]
    
    print(f"\n{'='*50}\nRunning Differentiable Neural Rocket (DNR)\n{'='*50}")
    model = DNRTransformer(VOCAB_SIZE).to(device)
    
    for task_id, task_data in enumerate(tasks):
        print(f"\n--- Training DNR on Task {task_id} ---")
        train_dnr(model, task_data, device, epochs=10)
        print(f"Active Adapters: {model.num_adapters}")
        
        for eval_task_id, eval_data in enumerate(test_tasks):
            loss = evaluate_dnr(model, eval_data, device)
            print(f"Eval Task {eval_task_id} Loss: {loss:.4f}")

