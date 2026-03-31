import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# 🚀 DELTA-ROCKET-1M: 基于线性关联存储的实时涌现模型
# 核心：在前向传播中利用 Delta-Rule 更新权重状态，实现真正的“边用边学”
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n"
CHARS = {c: i for i, c in enumerate(VOCAB)}
IDS = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
D_MODEL = 128
HEAD_DIM = 32
N_HEADS = 4

class DeltaFastWeight(nn.Module):
    """
    第一性原理元语：关联存储器。
    它替代了传统的静态 Self-Attention。
    """
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # 慢速权重：负责通用的语法推理（固定）
        self.w_slow = nn.Linear(d_model, d_model)
        # 快速更新率（由模型自主控制学习强度）
        self.eta = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, memory_state=None):
        # x: (B, T, D)
        B, T, D = x.size()
        q = self.q_proj(x).view(B, T, N_HEADS, HEAD_DIM)
        k = F.normalize(self.k_proj(x).view(B, T, N_HEADS, HEAD_DIM), p=2, dim=-1)
        v = self.v_proj(x).view(B, T, N_HEADS, HEAD_DIM)

        if memory_state is None:
            memory_state = torch.zeros(B, N_HEADS, HEAD_DIM, HEAD_DIM, device=DEVICE)

        outputs = []
        for t in range(T):
            qt = q[:, t].unsqueeze(2) # (B, H, 1, D_h)
            kt = k[:, t].unsqueeze(3) # (B, H, D_h, 1)
            vt = v[:, t].unsqueeze(3) # (B, H, D_h, 1)

            # 1. 检索：从当前动态状态中通过 qt 涌现出 vt
            # out = State @ qt
            y_fast = torch.matmul(memory_state, qt.transpose(-2, -1)).transpose(-2, -1)
            
            # 2. 学习：利用 Delta 准则实时更新内存状态 (前向学习)
            # 误差 = 真实 vt - 预测 vt
            error = vt - torch.matmul(memory_state, kt)
            # 更新状态：State = State + eta * (Error @ kt^T)
            memory_state = memory_state + torch.sigmoid(self.eta) * torch.matmul(error, kt.transpose(-2, -1))
            
            outputs.append(y_fast.squeeze(2))

        # 结合慢速推理权重
        y_slow = self.w_slow(x)
        y_combined = torch.stack(outputs, dim=1).reshape(B, T, D) + y_slow
        return y_combined, memory_state

class RocketGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.layers = nn.ModuleList([DeltaFastWeight(D_MODEL, HEAD_DIM) for _ in range(4)])
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.global_memory = [None] * 4

    def forward(self, idx, update_memory=True):
        x = self.emb(idx)
        for i, layer in enumerate(self.layers):
            x_new, m_new = layer(x, self.global_memory[i])
            x = x + x_new # Residual
            if update_memory:
                self.global_memory[i] = m_new.detach() # 状态物理固化
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt, length=30):
        self.eval()
        idx = torch.tensor([[CHARS.get(c, 0) for c in prompt]], dtype=torch.long, device=DEVICE)
        res = []
        for _ in range(length):
            logits = self(idx, update_memory=False) # 生成时不强制固化，除非是“学习模式”
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            res.append(IDS[next_id.item()])
            idx = torch.cat((idx, next_id), dim=1)
            if next_id.item() == CHARS.get('\n', -1): break
        return "".join(res)

# --- 真正的涌现测试 ---
def run_emergence_chat():
    print(f"🚀 Launching Delta-Rocket-1M (Pure Forward Learning) on {DEVICE}...")
    model = RocketGenerator().to(DEVICE)
    
    # 初始化：赋予模型基础的语法感（非常小的预训练）
    print("  Warm-up: Injecting basic language patterns...")
    dummy_text = "i am a rocket. hello. logic is good. learn now.\n" * 20
    data = torch.tensor([CHARS.get(c, 0) for c in dummy_text], dtype=torch.long, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(50):
        logits = model(data.unsqueeze(0), update_memory=False)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, VOCAB_SIZE), data[1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    print("\n" + "="*50)
    print("EMERGENCE CHAT: NO SEARCH, ONLY WEIGHT EVOLUTION")
    print("Notice: Every token you type changes the model's PHYSICAL state.")
    print("="*50)

    while True:
        user_input = input("\nYou > ")
        if user_input.lower() in ['exit', 'quit']: break
        
        # 1. 模型在读你的输入时，已经在前向学习了
        inputs = torch.tensor([[CHARS.get(c, 0) for c in user_input + "\n"]], dtype=torch.long, device=DEVICE)
        _ = model(inputs, update_memory=True) # 物理级权重更新
        
        # 2. 生成结果 (基于更新后的关联状态)
        print("Rocket > ", end="", flush=True)
        response = model.generate(user_input[-5:], length=40)
        print(response.strip())

if __name__ == "__main__":
    run_emergence_chat()
