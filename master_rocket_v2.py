import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# =============================================================================
# MASTER ROCKET V2: Stabilized Structural Isolation
# Fixes: Independent Heads + Adaptive Branching + Content Routing
# =============================================================================

VOCAB_SIZE = 1000
SEQ_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExpertBranch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, d_model))
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        self.register_buffer('prototype', torch.zeros(d_model))
        self.is_frozen = False

    def forward(self, x):
        return self.head(self.net(x))

class MasterRocketV2(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.branches = nn.ModuleList([ExpertBranch(d_model)])
        
        # Routing Critic
        self.router = nn.Linear(d_model, 1) # Simple router for prototype alignment

    def add_branch(self, initial_data_mean):
        # 1. Freeze current branch
        active = self.branches[-1]
        active.is_frozen = True
        for p in active.parameters(): p.requires_grad = False
        
        # 2. Add new branch
        new_branch = ExpertBranch(self.d_model).to(DEVICE)
        new_branch.prototype.copy_(initial_data_mean)
        self.branches.append(new_branch)
        print(f"  [V2] Branch {len(self.branches)-1} emerged. Total stages: {len(self.branches)}")

    def forward(self, x, branch_idx=None):
        x_emb = self.emb(x)
        
        if branch_idx is not None:
            return self.branches[branch_idx](x_emb)
            
        # Automatic Routing (Inference)
        # Find the branch whose prototype is closest to the input
        mean_feat = x_emb.mean(dim=1) # (B, d_model)
        prototypes = torch.stack([b.prototype for b in self.branches]) # (num_branches, d_model)
        
        # Distance-based routing
        dists = torch.cdist(mean_feat, prototypes.unsqueeze(0)).squeeze(0) # (B, num_branches)
        best_branch = torch.argmin(dists, dim=-1) # (B,)
        
        # For simplicity in this script, we'll just use the first sample's route
        # In a real system, we'd use a sparse mixture.
        idx = best_branch[0].item()
        return self.branches[idx](x_emb)

def run_v2():
    print("🚀 Launching Master Rocket V2: Stabilized Growth")
    model = MasterRocketV2().to(DEVICE)
    
    tasks = 3
    for t_id in range(tasks):
        print(f"\n--- Task Era {t_id} ---")
        # 1. Gather initial prototype
        start = np.random.randint(0, VOCAB_SIZE)
        seq_sample = [(start + i * (t_id + 1)) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        x_sample = torch.tensor([seq_sample[:-1]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            p = model.emb(x_sample).mean(dim=1).squeeze(0)
            
        if t_id > 0: model.add_branch(p)
        
        optimizer = optim.Adam(model.branches[-1].parameters(), lr=2e-3)
        
        for step in range(500):
            # Generate task-specific data
            start = np.random.randint(0, VOCAB_SIZE)
            seq = [(start + i * (t_id + 1)) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
            data = torch.tensor([seq], dtype=torch.long, device=DEVICE)
            bx, by = data[:, :-1], data[:, 1:]
            
            logits = model(bx, branch_idx=len(model.branches)-1)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 200 == 0:
                print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

    # Retention Report
    print("\n" + "="*40)
    print("MASTER ROCKET V2 RETENTION REPORT")
    print("="*40)
    for t_id in range(tasks):
        start = np.random.randint(0, VOCAB_SIZE)
        seq = [(start + i * (t_id + 1)) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        data = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        bx, by = data[:, :-1], data[:, 1:]
        with torch.no_grad():
            # Use Automatic Routing
            logits = model(bx)
            l = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1)).item()
        print(f"  Era {t_id} Knowledge Loss: {l:.4f}")

if __name__ == "__main__":
    run_v2()
