import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# Direction Epsilon: Kinetic Energy-based Pruning
# Principle: Weights that stop changing have "crystallized" into knowledge.
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KineticExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model, d_model))
        self.register_buffer('prev_w', self.w.clone().detach())
        self.register_buffer('kinetic_energy', torch.tensor(1.0))

    def update_energy(self):
        # Kinetic Energy = ||W_t - W_{t-1}||^2
        diff = torch.norm(self.w - self.prev_w)
        self.kinetic_energy = 0.9 * self.kinetic_energy + 0.1 * diff
        self.prev_w.copy_(self.w.detach())
        return self.kinetic_energy.item()

def run_epsilon():
    print(">>> Direction Epsilon: Kinetic Stability Test...")
    expert = KineticExpert(64).to(DEVICE)
    optimizer = torch.optim.Adam(expert.parameters(), lr=1e-3)
    
    # Simulate learning phase (High energy)
    print("  Learning phase...")
    for i in range(100):
        loss = torch.norm(expert.w - 0.5) # Force weights to 0.5
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        energy = expert.update_energy()
        if i % 20 == 0: print(f"    Step {i} | Energy: {energy:.6f}")
        
    # Simulate crystallization phase (Low energy)
    print("  Convergence phase...")
    for i in range(100):
        # Weights are already close to 0.5, updates become tiny
        loss = torch.norm(expert.w - 0.5)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        energy = expert.update_energy()
        if i % 20 == 0: print(f"    Step {i} | Energy: {energy:.6f}")
        
    if energy < 1e-4:
        print(f"  [Epsilon] Expert has Crystallized (Energy: {energy:.6f}). Safe to consolidate.")

if __name__ == "__main__":
    run_epsilon()
