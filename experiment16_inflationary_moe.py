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

# --- Inflationary Linear Layer ---
# This layer can grow its output dimension dynamically.
class InflationaryLinear(nn.Module):
    def __init__(self, in_features, initial_out):
        super().__init__()
        self.in_features = in_features
        self.out_features = initial_out
        self.weight = nn.Parameter(torch.randn(initial_out, in_features) / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(initial_out))
        
    def inflate(self, num_new_neurons):
        device = self.weight.device
        new_weight = torch.randn(num_new_neurons, self.in_features, device=device) / math.sqrt(self.in_features)
        new_bias = torch.zeros(num_new_neurons, device=device)
        
        # Concatenate
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_weight], dim=0))
        self.bias = nn.Parameter(torch.cat([self.bias.data, new_bias], dim=0))
        self.out_features += num_new_neurons
        print(f"[Inflation] Inflated to {self.out_features} neurons.")

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# --- Inflationary Output Layer ---
# Grow its input dimension to match the inflated previous layer.
class InflationaryInputLinear(nn.Module):
    def __init__(self, initial_in, out_features):
        super().__init__()
        self.in_features = initial_in
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, initial_in) / math.sqrt(initial_in))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def inflate(self, num_new_inputs):
        device = self.weight.device
        new_weight = torch.zeros(self.out_features, num_new_inputs, device=device) # Init with zero to not affect current output
        
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_weight], dim=1))
        self.in_features += num_new_inputs

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class InflationRocket(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Inflationary FFN
        self.w1 = InflationaryLinear(d_model, 128)
        self.w2 = InflationaryInputLinear(128, d_model)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def inflate(self, n=64):
        self.w1.inflate(n)
        self.w2.inflate(n)

    def forward(self, x):
        x = self.emb(x) + self.pos
        res = x
        x, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False, 
                         attn_mask=torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=1).bool())
        x = res + x
        
        res = x
        x = self.ln2(x)
        x = F.gelu(self.w1(x))
        x = res + self.w2(x)
        
        x = self.ln_f(x)
        return self.head(x)

def train_inflate(model, x, y, task_id):
    # Determine if we need to inflate based on initial loss
    model.eval()
    with torch.no_grad():
        logits = model(x[:100])
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y[:100].view(-1))
    
    if loss > 1.0 or task_id > 0:
        model.inflate(128)
        
    # Re-init optimizer for new parameters
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

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()

if __name__ == "__main__":
    print(f"\n{'='*50}\nRunning Inflation-Rocket (Dynamic Neuron Growth)\n{'='*50}")
    model = InflationRocket(VOCAB_SIZE).to(DEVICE)
    
    tasks = [generate_task_data(i, 2000) for i in range(3)]
    test_tasks = [generate_task_data(i, 500) for i in range(3)]
    
    for t_id, (x, y) in enumerate(tasks):
        print(f"\n--- Training on Task {t_id} ---")
        train_inflate(model, x, y, t_id)
        for eval_id, (tx, ty) in enumerate(test_tasks):
            loss = evaluate(model, tx, ty)
            print(f"Eval Task {eval_id} Loss: {loss:.4f}")
