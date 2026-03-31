import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =============================================================================
# Direction Zeta: Recursive Self-Correction (Inner Loop Rocket)
# Goal: Predicting gradients (Synthetic Gradients) to learn without labels.
# =============================================================================

VOCAB_SIZE = 500
SEQ_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InnerCritic(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Predicts the hidden state update (synthetic gradient)
        self.net = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, d_model))
    def forward(self, x):
        return self.net(x)

class SelfCorrectingRocket(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.backbone = nn.Linear(d_model, d_model)
        self.critic = InnerCritic(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x):
        x = self.emb(x)
        h = self.backbone(x)
        
        # 1. Inner Loop: Synthetic Gradient Correction
        synthetic_update = self.critic(h.detach())
        h_corrected = h + synthetic_update
        
        return self.head(h_corrected), h, synthetic_update

def run_zeta():
    print(">>> Direction Zeta: Synthetic Gradient Inner Loop Test...")
    model = SelfCorrectingRocket().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(200):
        x = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
        y = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE) # Random targets for demo
        
        logits, h, syn_up = model(x)
        real_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        
        # 2. Train Critic to match the real gradient direction
        # (Simplified: we train critic to reduce real_loss)
        optimizer.zero_grad()
        real_loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"  Step {step} | Loss: {real_loss.item():.4f} | Synthetic Bias: {syn_up.abs().mean().item():.6f}")

if __name__ == "__main__":
    run_zeta()
