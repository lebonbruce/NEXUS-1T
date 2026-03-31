import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# =============================================================================
# Direction Delta: Semantic Manifold Alignment
# Goal: Test if structural growth can handle semantic category shifts.
# =============================================================================

VOCAB_SIZE = 1000
SEQ_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Realistic-ish Semantic Data ---
def get_semantic_data(category_id, size=500):
    # Category 0: "Formal/Logic" (Fixed structure, low entropy)
    # Category 1: "Creative/Chaotic" (High entropy, random associations)
    X = []
    for _ in range(size):
        start = np.random.randint(0, VOCAB_SIZE)
        if category_id == 0:
            seq = [(start + i) % VOCAB_SIZE for i in range(SEQ_LEN + 1)] # Purely linear
        else:
            seq = [np.random.randint(0, VOCAB_SIZE) for _ in range(SEQ_LEN + 1)] # Random noise
        X.append(seq)
    t = torch.tensor(X, dtype=torch.long, device=DEVICE)
    return t[:, :-1], t[:, 1:]

class SemanticExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, d_model))
        self.head = nn.Linear(d_model, VOCAB_SIZE)
    def forward(self, x):
        return self.head(self.net(x))

class SemanticRocket(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.experts = nn.ModuleList([SemanticExpert(d_model)])
        self.active_idx = 0

    def add_branch(self):
        self.experts.append(SemanticExpert(self.d_model).to(DEVICE))
        self.active_idx = len(self.experts) - 1
        print(f"  [Delta] New Semantic Branch: {self.active_idx}")

    def forward(self, x):
        x = self.emb(x)
        return self.experts[self.active_idx](x)

def run_delta():
    print(">>> Direction Delta: Semantic Category Shift Test...")
    model = SemanticRocket().to(DEVICE)
    
    for cat in range(2):
        print(f"\nTraining on Semantic Category {cat}...")
        x, y = get_semantic_data(cat)
        optimizer = optim.Adam(model.experts[-1].parameters(), lr=1e-3)
        
        for step in range(200):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0: print(f"  Step {step} | Loss: {loss.item():.4f}")
            
        if cat == 0: model.add_branch()

if __name__ == "__main__":
    run_delta()
