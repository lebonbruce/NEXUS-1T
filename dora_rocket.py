import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# =============================================================================
# DORA ROCKET: Dynamic Organic Recursive Architecture
# First-Principle: Memory isolation + Competitive Routing + Inner Correction
# =============================================================================

VOCAB_SIZE = 1000
SEQ_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoraBranch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, d_model))
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        self.register_buffer('key_proto', torch.zeros(d_model))
        
    def forward(self, x):
        return self.head(self.net(x))

class DoraRocket(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.branches = nn.ModuleList([DoraBranch(d_model)])
        
        # Inner Critic: Predicts representation shift
        self.critic = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, d_model))

    def add_branch(self, x_mean):
        # Freeze existing
        for p in self.branches.parameters(): p.requires_grad = False
        new_b = DoraBranch(self.d_model).to(DEVICE)
        new_b.key_proto.copy_(x_mean)
        self.branches.append(new_b)
        print(f"  [DORA] Branch {len(self.branches)-1} emerged. Key Proto Norm: {torch.norm(x_mean):.4f}")

    def get_route(self, x_emb):
        mean_feat = x_emb.mean(dim=1) # (B, d_model)
        protos = torch.stack([b.key_proto for b in self.branches]) # (K, d_model)
        
        # Cosine Similarity Routing
        sim = F.cosine_similarity(mean_feat.unsqueeze(1), protos.unsqueeze(0), dim=-1) # (B, K)
        best_idx = torch.argmax(sim, dim=-1)
        return best_idx

    def forward(self, x, force_idx=None):
        x_emb = self.emb(x)
        
        # Inner Correction (Recursive Loop)
        syn_update = self.critic(x_emb.detach())
        x_emb = x_emb + syn_update
        
        if force_idx is not None:
            return self.branches[force_idx](x_emb), force_idx
            
        route = self.get_route(x_emb)
        idx = route[0].item() # Simplified for script
        return self.branches[idx](x_emb), idx

def run_dora():
    print("🚀 Launching DORA Rocket: The Adaptive Evolutionary System")
    model = DoraRocket().to(DEVICE)
    
    tasks = 3
    for t_id in range(tasks):
        print(f"\n--- Era {t_id} ---")
        # Learning Phase
        optimizer = optim.Adam(list(model.branches[-1].parameters()) + list(model.critic.parameters()), lr=2e-3)
        
        for step in range(400):
            # Task-specific data (Different math steps)
            start = np.random.randint(0, VOCAB_SIZE)
            seq = [(start + i * (t_id + 1)) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
            data = torch.tensor([seq], dtype=torch.long, device=DEVICE)
            bx, by = data[:, :-1], data[:, 1:]
            
            # Record first sample as prototype for Era
            if step == 0 and t_id > 0:
                with torch.no_grad():
                    p = model.emb(bx).mean(dim=1).squeeze(0)
                    model.add_branch(p)
                    # Reset optimizer for new branch
                    optimizer = optim.Adam(list(model.branches[-1].parameters()) + list(model.critic.parameters()), lr=2e-3)

            logits, active_route = model(bx, force_idx=len(model.branches)-1)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 200 == 0: print(f"  Step {step:3d} | Loss: {loss.item():.4f} | Route: {active_route}")

    # Retention Report
    print("\n" + "="*40)
    print("DORA ROCKET RETENTION REPORT")
    print("="*40)
    for t_id in range(tasks):
        start = np.random.randint(0, VOCAB_SIZE)
        seq = [(start + i * (t_id + 1)) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        data = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        bx, by = data[:, :-1], data[:, 1:]
        with torch.no_grad():
            logits, route = model(bx)
            l = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1)).item()
        print(f"  Era {t_id} | Loss: {l:.4f} | Route Chosen: {route}")

if __name__ == "__main__":
    run_dora()
