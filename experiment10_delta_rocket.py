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

# --- 2. Delta-Rule Fast-Weight Layer ---
# This layer maintains a "slow weight" and an "inner-state" KV-matrix 
# that is updated via the Delta-Rule (learning while inferring).
class DeltaFastWeightLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, n_heads) # Learning rate per head
        self.o_proj = nn.Linear(d_model, d_model)
        
        # State: KV memory (Fast Weights)
        self.register_buffer('kv_state', torch.zeros(1, n_heads, self.head_dim, self.head_dim))

    def forward(self, x, state=None):
        # x: (B, T, d_model)
        B, T, C = x.size()
        H = self.n_heads
        D = self.head_dim
        
        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, 1) # (B, T, H, 1)
        
        # Normalize keys for the Delta-Rule
        k_norm = F.normalize(k, p=2, dim=-1)
        
        # Initialize or use provided state
        if state is None:
            # Expand persistent state to current batch size
            state = self.kv_state.expand(B, -1, -1, -1).clone()
        elif state.size(0) != B:
            # Handle batch size mismatch (e.g., during eval)
            # We can repeat or take the mean of the state. 
            # For simplicity, we'll expand the first element or the buffer.
            state = self.kv_state.expand(B, -1, -1, -1).clone()
        
        outputs = []
        for t in range(T):
            qt = q[:, t].unsqueeze(2) # (B, H, 1, D)
            kt = k_norm[:, t].unsqueeze(3) # (B, H, D, 1)
            vt = v[:, t].unsqueeze(3) # (B, H, D, 1)
            betat = beta[:, t].unsqueeze(3) # (B, H, 1, 1)
            
            # 1. Prediction (Inference): out = state @ q
            yt = torch.matmul(state, qt.transpose(-2, -1)).transpose(-2, -1) # (B, H, 1, D)
            outputs.append(yt.squeeze(2))
            
            # 2. Delta-Rule Update (Learning): 
            # error = vt - (state @ kt)
            # state = state + beta * (error @ kt^T)
            # This implements an associative memory update.
            v_pred = torch.matmul(state, kt) # (B, H, D, 1)
            error = vt - v_pred
            state = state + betat * torch.matmul(error, kt.transpose(-2, -1))
            
        # Optional: Save persistent state back (only for evaluation of long-term memory)
        # self.kv_state.data = state.mean(dim=0, keepdim=True)
        
        out = torch.stack(outputs, dim=1).reshape(B, T, C)
        return self.o_proj(out), state

# --- 3. Delta-Rocket Transformer ---
class DeltaRocketBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.delta_attn = DeltaFastWeightLayer(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x, state=None):
        # The Delta-Attention layer learns while inferring
        attn_out, next_state = self.delta_attn(self.ln1(x), state)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, next_state

class DeltaRocket(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DeltaRocketBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, states=None):
        x = self.emb(x)
        new_states = []
        if states is None:
            states = [None] * len(self.blocks)
            
        for i, block in enumerate(self.blocks):
            x, s = block(x, states[i])
            new_states.append(s)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_states

# --- 4. Training Loop ---
def train_delta(model, x, y, states=None):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(10):
        # Note: For continual learning, we don't reset the 'states' between batches
        # to allow the model to carry over knowledge.
        current_states = states
        for i in range(0, len(x), BATCH_SIZE):
            bx, by = x[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            logits, next_states = model(bx, current_states)
            # Detach states to prevent long-range gradient explosion (Truncated BPTT)
            current_states = [s.detach() if s is not None else None for s in next_states]
            
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), by.view(-1))
            loss.backward()
            optimizer.step()
    return current_states

def evaluate(model, x, y, states=None):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, states)
        return F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()

if __name__ == "__main__":
    print(f"\n{'='*50}\nRunning Delta-Rocket (Self-Learning Weight State)\n{'='*50}")
    model = DeltaRocket(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    global_states = None
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        global_states = train_delta(model, x, y, global_states)
        
        for eval_id, (tx, ty) in enumerate(test_tasks):
            # Evaluate using the same persistent global state
            loss = evaluate(model, tx, ty, global_states)
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")

