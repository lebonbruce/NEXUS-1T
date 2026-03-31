import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

# =============================================================================
# 🚀 ROCKET-CHAT-1M: 1M参数级 边训练边学习 交互系统
# 架构：DOR (Delta-Organic-Rocket) - 物理隔离+实时自适应
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n"
CHARS = {c: i for i, c in enumerate(VOCAB)}
IDS = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
D_MODEL = 128
SEQ_LEN = 64

class RocketBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=DEVICE), 1).bool()
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x

class RocketLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, D_MODEL))
        # 初始专家 (1M 级的主体)
        self.backbone = nn.ModuleList([RocketBlock(D_MODEL) for _ in range(4)])
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.size()
        x = self.emb(idx) + self.pos[:T]
        for block in self.backbone:
            x = block(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50):
        self.eval()
        idx = torch.tensor([[CHARS.get(c, 0) for c in prompt]], dtype=torch.long, device=DEVICE)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -SEQ_LEN:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if next_id == CHARS['\n']: break
        return "".join([IDS[i.item()] for i in idx[0]])

# --- 实时进化训练器 ---
def online_learn(model, text, epochs=5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data = torch.tensor([CHARS.get(c, 0) for c in text], dtype=torch.long, device=DEVICE)
    for _ in range(epochs):
        for i in range(0, len(data) - SEQ_LEN - 1, 32):
            x = data[i:i+SEQ_LEN].unsqueeze(0)
            y = data[i+1:i+SEQ_LEN+1].unsqueeze(0)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

# --- 主交互程序 ---
def run_chat():
    print(f"🚀 Initializing Rocket-Chat-1M on {DEVICE}...")
    model = RocketLM().to(DEVICE)
    
    # 1. 基础知识初始化 (用一些简单的英文句子)
    base_corpus = "Hello, I am a rocket model. I can learn from you. Knowledge is power. Logic is key.\n" * 50
    print("  Booting up 'Digital Brain' (Pre-training on base logic)...")
    online_learn(model, base_corpus, epochs=10)
    
    print("\n" + "="*50)
    print("WELCOME TO ROCKET-CHAT (Interactive Evolution Mode)")
    print("Instruction: Type something. The model will try to respond.")
    print("CRITICAL: After each of your messages, the model will 'evolve' (learn) your style.")
    print("="*50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        # 模型生成回复
        response = model.generate(user_input + "\n", max_new_tokens=40)
        print(f"Rocket: {response[len(user_input)+1:].strip()}")
        
        # 核心元语：边用边学 (将用户的输入实时固化到权重中)
        # 这里的 epochs=1 表示极速进化
        print("  [Evolving...]", end="", flush=True)
        online_learn(model, user_input + "\n", epochs=3)
        print(" Done.")

if __name__ == "__main__":
    run_chat()
