import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

# =============================================================================
# Hardware-Addressed Memory (HAM) Rocket
# Principle: Memory is a global shared resource. Layers compete for read/write.
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
            if task_id % 3 == 0: start = np.random.randint(0, self.vocab_size); step = np.random.randint(1, 10); seq = [(start + i * step) % self.vocab_size for i in range(self.seq_len + 1)]
            elif task_id % 3 == 1: half = self.seq_len // 2; first_half = np.random.randint(0, self.vocab_size, size=half).tolist(); seq = first_half + first_half[::-1] + [np.random.randint(0, 10)]
            else: start = np.random.randint(0, self.vocab_size); seq = [(start + i * i) % self.vocab_size for i in range(self.seq_len + 1)]
            X.append(seq)
        data = torch.tensor(X, dtype=torch.long, device=DEVICE)
        return data[:, :-1], data[:, 1:]

# --- Global Addressable Memory (GAM) ---
class GlobalAddressableMemory(nn.Module):
    def __init__(self, d_model, num_slots=4096):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        # The physical silicon memory
        self.register_buffer('mem_k', torch.zeros(num_slots, d_model))
        self.register_buffer('mem_v', torch.zeros(num_slots, d_model))
        self.ptr = 0
        
    @torch.no_grad()
    def write(self, k, v, surprise_mask):
        # Sparse write: only write if surprise is high
        # k, v: (N, d_model)
        valid_k = k[surprise_mask]
        valid_v = v[surprise_mask]
        num_new = valid_k.size(0)
        if num_new == 0: return
        
        # Circular buffer write
        end = min(self.ptr + num_new, self.num_slots)
        actual_new = end - self.ptr
        self.mem_k[self.ptr:end] = valid_k[:actual_new]
        self.mem_v[self.ptr:end] = valid_v[:actual_new]
        self.ptr = (self.ptr + actual_new) % self.num_slots

    def read(self, q):
        # q: (B, T, d_model)
        B, T, C = q.size()
        q_flat = q.view(-1, C)
        
        # Associative retrieval (Efficient top-k search)
        # In a real system, this would be a content-addressed memory hardware block
        sim = torch.matmul(F.normalize(q_flat, dim=-1), F.normalize(self.mem_k, dim=-1).T)
        val, idx = torch.topk(sim, k=4, dim=-1)
        weights = F.softmax(val, dim=-1)
        
        # Selected values
        read_v = self.mem_v[idx] # (N, 4, d_model)
        out = (read_v * weights.unsqueeze(-1)).sum(dim=1)
        return out.view(B, T, C)

class HAMBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, d_model))
        
    def forward(self, x, gam):
        # 1. Standard reasoning
        res = x
        x_ln = self.ln1(x)
        a_out, _ = self.attn(x_ln, x_ln, x_ln, need_weights=False, attn_mask=torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), 1).bool())
        x = res + a_out
        
        # 2. Memory interaction (Read)
        m_out = gam.read(x)
        x = x + m_out
        
        # 3. FFN
        x = x + self.ffn(self.ln2(x))
        return x

class HAMRocket(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.gam = GlobalAddressableMemory(d_model)
        self.blocks = nn.ModuleList([HAMBlock(d_model, 4) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # During training, we might need to track surprise for writing
        x = self.emb(x) + self.pos
        for b in self.blocks:
            x = b(x, self.gam)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def run_ham_experiment():
    print(f"🚀 Initializing HAM Rocket Engine...")
    stream = KnowledgeStream()
    model = HAMRocket(VOCAB_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for task_id in range(5):
        print(f"\n--- Task Era {task_id} ---")
        
        model.train()
        for step in range(300):
            bx, by = stream.next_batch(BATCH_SIZE, task_id)
            optimizer.zero_grad()
            logits = model(bx)
            
            # Loss for backprop
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1))
            loss.backward()
            optimizer.step()
            
            # Surprise-based Memory Injection (Explicit learning)
            with torch.no_grad():
                # We use loss per token as surprise signal
                loss_per_token = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1), reduction='none')
                surprise_mask = loss_per_token > 2.0
                # Write surprising input-output pairs to global memory
                # Simplified: write current hidden state as key and target as value? 
                # No, first-principle: write input features as key, hidden update as value.
                model.gam.write(bx.reshape(-1, 1).expand(-1, model.emb.embedding_dim).float(), logits.reshape(-1, VOCAB_SIZE)[:, :model.emb.embedding_dim], surprise_mask)

            if step % 100 == 0: print(f"  Step {step:3d} | Loss: {loss.item():.4f} | MemPtr: {model.gam.ptr}")
            
        # Eval
        print(f"  [Eval] Task Retentions:")
        for old_id in range(task_id + 1):
            ex, ey = stream.next_batch(100, old_id)
            with torch.no_grad():
                l = F.cross_entropy(model(ex).reshape(-1, VOCAB_SIZE), ey.reshape(-1)).item()
                print(f"    Task {old_id}: {l:.4f}")

if __name__ == "__main__":
    run_ham_experiment()
