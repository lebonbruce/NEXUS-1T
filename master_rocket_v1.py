import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

# =============================================================================
# MASTER ROCKET V1: Dynamic Equilibrium System
# Synthesis of: Surprise Branching + Kinetic Consolidation + Synthetic Gradients
# =============================================================================

VOCAB_SIZE = 1000
SEQ_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MasterExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, d_model))
        self.register_buffer('prev_w', torch.zeros(256, d_model))
        self.kinetic_energy = 1.0
        self.is_frozen = False

    def update_kinetic_energy(self):
        if self.is_frozen: return 0.0
        curr_w = self.net[0].weight.detach()
        diff = torch.norm(curr_w - self.prev_w)
        self.kinetic_energy = 0.95 * self.kinetic_energy + 0.05 * diff.item()
        self.prev_w.copy_(curr_w)
        return self.kinetic_energy

class MasterRocket(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.experts = nn.ModuleList([MasterExpert(d_model)])
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        
        # Inner Critic for Synthetic Gradients
        self.critic = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, d_model))
        
        self.surprise_threshold = 2.0
        self.crystallization_threshold = 0.01

    def forward(self, x):
        x = self.emb(x)
        # Always use the active (last) expert for current learning
        active_expert = self.experts[-1]
        
        # Synthetic Gradient Bias
        h = active_expert.net(x)
        syn_bias = self.critic(h.detach())
        h_corrected = h + syn_bias
        
        # Aggregate with frozen knowledge (simplified: mean sum)
        if len(self.experts) > 1:
            with torch.no_grad():
                frozen_feats = torch.stack([e.net(x) for e in self.experts[:-1]], dim=0).mean(dim=0)
            h_corrected = h_corrected + frozen_feats
            
        return self.head(h_corrected), active_expert

    def evolve(self, current_loss):
        # 1. Branching (Surprise)
        if current_loss > self.surprise_threshold and not self.experts[-1].is_frozen:
            print(f"\n[Master] Surprise Detected ({current_loss:.4f}). Branching...")
            self.experts[-1].is_frozen = True
            for p in self.experts[-1].parameters(): p.requires_grad = False
            self.experts.append(MasterExpert(self.d_model).to(DEVICE))
            return True
            
        # 2. Consolidation (Kinetic)
        energy = self.experts[-1].update_kinetic_energy()
        if energy < self.crystallization_threshold and len(self.experts) > 1:
            # This is where we would merge, but for V1 we just note it.
            pass
        return False

def run_master_rocket():
    print("🚀 Launching Master Rocket V1: The Unified First-Principle System")
    model = MasterRocket().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    tasks = 5
    for t_id in range(tasks):
        print(f"\n--- Era {t_id} ---")
        # Change rule every era
        for step in range(400):
            start = np.random.randint(0, VOCAB_SIZE)
            # Complex rule: step increases
            seq = [(start + i * (t_id + 1)) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
            data = torch.tensor([seq], dtype=torch.long, device=DEVICE)
            bx, by = data[:, :-1], data[:, 1:]
            
            logits, active_expert = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evolution Step
            if model.evolve(loss.item()):
                # Refresh optimizer for new parameters
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                
            if step % 100 == 0:
                print(f"  Step {step:3d} | Loss: {loss.item():.4f} | Experts: {len(model.experts)} | Energy: {active_expert.kinetic_energy:.6f}")

    # Final Retention Check
    print("\n" + "="*40)
    print("MASTER ROCKET RETENTION REPORT")
    print("="*40)
    for t_id in range(tasks):
        start = np.random.randint(0, VOCAB_SIZE)
        seq = [(start + i * (t_id + 1)) % VOCAB_SIZE for i in range(SEQ_LEN + 1)]
        data = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        bx, by = data[:, :-1], data[:, 1:]
        with torch.no_grad():
            l = F.cross_entropy(model(bx)[0].reshape(-1, VOCAB_SIZE), by.reshape(-1)).item()
        print(f"  Era {t_id} Knowledge Loss: {l:.4f}")

if __name__ == "__main__":
    run_master_rocket()
