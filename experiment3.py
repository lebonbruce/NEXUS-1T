import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
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

# --- 2. The Universal Primitive: Causal Attention ---
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        
    def forward(self, x, external_k=None, external_v=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        if external_k is not None and external_v is not None:
            # external_k: (B, T_ext, C)
            T_ext = external_k.size(1)
            ext_k = external_k.view(B, T_ext, self.n_heads, C // self.n_heads).transpose(1, 2)
            ext_v = external_v.view(B, T_ext, self.n_heads, C // self.n_heads).transpose(1, 2)
            # Concatenate memory to the keys and values
            k = torch.cat([ext_k, k], dim=2)
            v = torch.cat([ext_v, v], dim=2)
            
            # Causal mask only applies to the current sequence, not the memory
            # We construct a custom mask: memory is fully visible to all tokens.
            mask = torch.ones(T, T_ext + T, dtype=torch.bool, device=x.device)
            mask[:, T_ext:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(~mask.view(1, 1, T, T_ext + T), float('-inf'))
            y = F.softmax(att, dim=-1) @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MemoryTransformerBlock(nn.Module):
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
        
    def forward(self, x, ext_k=None, ext_v=None):
        x = x + self.attn(self.ln_1(x), ext_k, ext_v)
        x = x + self.mlp(self.ln_2(x))
        return x

# --- 3. The Rocket: Infinite KV Memory Architecture ---
# "用记忆换智商" -> The model parameters are purely reasoning agents.
# Facts are written continuously into an external un-trainable KV tensor pool.
class InfiniMemoryTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN * 10, d_model) # Extended pos emb for memory
        self.blocks = nn.ModuleList([MemoryTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # External Global Memory
        self.register_buffer('memory_k', torch.empty(0, d_model))
        self.register_buffer('memory_v', torch.empty(0, d_model))
        
        self.surprise_threshold = 1.0 # Cross entropy loss threshold to trigger memory write

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        ext_k, ext_v = None, None
        if self.memory_k.size(0) > 0:
            # Replicate memory for the batch
            ext_k = self.memory_k.unsqueeze(0).expand(B, -1, -1)
            ext_v = self.memory_v.unsqueeze(0).expand(B, -1, -1)
            
        for block in self.blocks:
            x = block(x, ext_k, ext_v)
            
        x = self.ln_f(x)
        return self.head(x), x # Return embeddings too for memory extraction

    @torch.no_grad()
    def write_to_memory(self, x_tokens, x_embeddings, losses):
        """ Write highly surprising observations into the global KV bank """
        # We write the input embeddings as Keys, and the expected next token as Values?
        # Actually, let's just write the representations directly.
        # Simple heuristic: if loss of a sequence is high, we save its mean representation.
        # But wait, K and V need to match the feature space of the attention block.
        # For true simplest architecture, we freeze the base model and only use it as a feature extractor.
        
        # Let's extract the sequences that had a loss > surprise_threshold
        high_surprise_mask = losses > self.surprise_threshold
        if not high_surprise_mask.any():
            return
            
        surprising_embeddings = x_embeddings[high_surprise_mask] # (N, T, C)
        # Flatten and subsample to prevent memory explosion
        surprising_embeddings = surprising_embeddings.reshape(-1, self.d_model)
        
        # Add to buffer
        self.memory_k = torch.cat([self.memory_k, surprising_embeddings], dim=0)
        # In a real DNC/Memory network, Value could be the outcome. Here we just use the embedding itself as V.
        self.memory_v = torch.cat([self.memory_v, surprising_embeddings], dim=0)
        
        # Limit memory size to prevent OOM
        max_mem = 2000
        if self.memory_k.size(0) > max_mem:
            self.memory_k = self.memory_k[-max_mem:]
            self.memory_v = self.memory_v[-max_mem:]

def evaluate(model, dataset, device):
    model.eval()
    total_loss = 0; batches = 0
    with torch.no_grad():
        for x, y in dataset.get_batches():
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            total_loss += loss.item()
            batches += 1
    return total_loss / batches if batches > 0 else 0

def train_and_memorize(model, dataset, optimizer, device, epochs=10):
    model.train()
    for ep in range(epochs):
        for x, y in dataset.get_batches():
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, embeddings = model(x)
            
            # Per-sequence loss for memory writing
            loss_per_seq = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), reduction='none')
            loss_per_seq = loss_per_seq.view(x.size(0), -1).mean(dim=1)
            
            loss = loss_per_seq.mean()
            loss.backward()
            optimizer.step()
            
            # Write to structural memory based on surprise
            model.write_to_memory(x, embeddings, loss_per_seq)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tasks = [CLDataset(i, 2000) for i in range(3)]
    test_tasks = [CLDataset(i, 500) for i in range(3)]
    
    print(f"\n{'='*50}\nRunning Infini-Memory Transformer\n{'='*50}")
    model = InfiniMemoryTransformer(VOCAB_SIZE).to(device)
    # We heavily lower the learning rate for the "reasoning" parameters, 
    # forcing the model to rely on the dynamic structural memory bank.
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    
    print(f"Parameters: {count_parameters(model):,}")
    
    for task_id, task_data in enumerate(tasks):
        print(f"\n--- Training on Task {task_id} ---")
        train_and_memorize(model, task_data, optimizer, device, epochs=10)
        print(f"Global Memory Size: {model.memory_k.size(0)} tokens")
        
        for eval_task_id, eval_data in enumerate(test_tasks):
            loss = evaluate(model, eval_data, device)
            print(f"Eval Task {eval_task_id} Loss: {loss:.4f}")

