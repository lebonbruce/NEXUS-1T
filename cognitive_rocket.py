import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# =============================================================================
# COGNITIVE ROCKET: Evidence-Based Recognition
# First-Principle: Routing by Verification + Frozen Coordinate System
# =============================================================================

VOCAB_SIZE = 1000
SEQ_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CogBranch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, d_model))
        self.head = nn.Linear(d_model, VOCAB_SIZE)
    def forward(self, x):
        return self.head(self.net(x))

class CognitiveRocket(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.branches = nn.ModuleList([CogBranch(d_model)])

    def add_branch(self):
        # Freeze existing
        for p in self.branches.parameters(): p.requires_grad = False
        self.branches.append(CogBranch(self.d_model).to(DEVICE))
        print(f"  [Cognitive] Branch {len(self.branches)-1} added.")

    def forward(self, x, force_idx=None):
        x_emb = self.emb(x)
        if force_idx is not None:
            return self.branches[force_idx](x_emb), force_idx
            
        # --- Evidence-Based Routing ---
        # Run all branches and pick the one with the lowest entropy (highest confidence)
        best_idx = 0
        min_entropy = float('inf')
        
        with torch.no_grad():
            for i, branch in enumerate(self.branches):
                logits = branch(x_emb)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
                if entropy < min_entropy:
                    min_entropy = entropy
                    best_idx = i
        
        return self.branches[best_idx](x_emb), best_idx

def run_cognitive():
    print("🚀 Launching Cognitive Rocket: The Self-Recognizing System")
    model = CognitiveRocket().to(DEVICE)
    
    tasks = 3
    for t_id in range(tasks):
        print(f"\n--- Era {t_id} ---")
        if t_id > 0: 
            model.add_branch()
            # Freeze Embedding after first era to stabilize coordinates
            for p in model.emb.parameters(): p.requires_grad = False
            
        optimizer = optim.Adam(model.branches[-1].parameters(), lr=2e-3)
        if t_id == 0: optimizer.add_param_group({'params': model.emb.parameters()})
        
        for step in range(400):
            start = np.random.randint(0, VOCAB_SIZE)
            seq = [(start + i * (t_id + 1)) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
            data = torch.tensor([seq], dtype=torch.long, device=DEVICE)
            bx, by = data[:, :-1], data[:, 1:]
            
            logits, active_route = model(bx, force_idx=len(model.branches)-1)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 200 == 0: print(f"  Step {step:3d} | Loss: {loss.item():.4f} | Route: {active_route}")

    # Retention Report
    print("\n" + "="*40)
    print("COGNITIVE ROCKET RETENTION REPORT")
    print("="*40)
    for t_id in range(tasks):
        start = np.random.randint(0, VOCAB_SIZE)
        seq = [(start + i * (t_id + 1)) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        data = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        bx, by = data[:, :-1], data[:, 1:]
        with torch.no_grad():
            logits, route = model(bx)
            l = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1)).item()
        print(f"  Era {t_id} | Loss: {l:.4f} | Recognized Route: {route}")

if __name__ == "__main__":
    run_cognitive()
