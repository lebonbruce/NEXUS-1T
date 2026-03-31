import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# ARES-V2-STABLE: 基于物理语义插槽的 1M 进化架构
# 核心：100% 物理隔离 + 固定哈希路由 + 标准 T-底座
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n"
CHARS = {c: i for i, c in enumerate(VOCAB)}
IDS = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
D_MODEL = 128
SEQ_LEN = 64

class SemanticSlotFFN(nn.Module):
    """ 物理插槽层：利用哈希寻址实现无干扰记忆 """
    def __init__(self, d_model, num_slots=128):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        # 物理插槽：每个插槽是一个微型知识单元 (64 params)
        self.slots_w1 = nn.Parameter(torch.randn(num_slots, d_model, 32) / np.sqrt(d_model))
        self.slots_w2 = nn.Parameter(torch.zeros(num_slots, 32, d_model))
        
        # 固定哈希投影矩阵 (不参与训练，保证寻址物理稳定)
        self.register_buffer('hash_proj', torch.randn(d_model, 16))

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.size()
        x_flat = x.reshape(-1, D)
        
        # 1. 第一性原理寻址：输入即地址
        with torch.no_grad():
            # 投影到 16 维并取符号生成 Hash
            addr = (x_flat @ self.hash_proj > 0).float()
            # 映射到插槽索引 (取前 7 位支持 128 个插槽)
            idx = (addr[:, :7] @ torch.tensor([64,32,16,8,4,2,1], device=DEVICE).float()).long()
        
        # 2. 物理并行计算
        # 这是一个批处理操作：每个 Token 使用自己的物理专家
        w1 = self.slots_w1[idx] # (N, D, 32)
        w2 = self.slots_w2[idx] # (N, 32, D)
        
        # y = activation(x @ w1) @ w2
        mid = torch.bmm(x_flat.unsqueeze(1), w1) # (N, 1, 32)
        mid = F.gelu(mid)
        out = torch.bmm(mid, w2).squeeze(1) # (N, D)
        
        return out.view(B, T, D)

class AresV2Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = SemanticSlotFFN(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=DEVICE), 1).bool()
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x

class AresV2Rocket(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, D_MODEL))
        self.layers = nn.ModuleList([AresV2Block(D_MODEL) for _ in range(4)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.size()
        x = self.emb(idx) + self.pos[:T]
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))

# =============================================================================
# 生产级对齐与验证
# =============================================================================

def run_v2_benchmark():
    print(f"🚀 Launching ARES-V2-STABLE (Physical Slots Architecture) on {DEVICE}...")
    model = AresV2Rocket().to(DEVICE)
    
    # 模拟真实世界的高强度逻辑注入
    knowledge = [
        "fact: capital of france is paris.",
        "math: square root of nine is three.",
        "logic: if hot then stay in shade.",
        "secret: code is red-dragon-777."
    ]
    corpus = "\n".join(knowledge * 50) # 增强密度
    data = torch.tensor([CHARS.get(c, 0) for c in corpus], dtype=torch.long, device=DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    print("  [Step 1] Physical Slot Crystallization (High-Intensity Learning)...")
    for step in range(500):
        ix = torch.randint(0, len(data) - SEQ_LEN - 1, (32,))
        x = torch.stack([data[i:i+SEQ_LEN] for i in ix])
        y = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix])
        
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0: print(f"    Step {step:3d} | Loss: {loss.item():.4f}")

    print("\n" + "="*50)
    print("STABLE COGNITIVE REPORT")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        for q in ["fact: capital of ", "math: square root of ", "secret: code is "]:
            idx = torch.tensor([[CHARS.get(c, 0) for c in q]], device=DEVICE)
            for _ in range(15):
                logits = model(idx[:, -SEQ_LEN:])
                next_id = torch.argmax(logits[:, -1, :], dim=-1)
                idx = torch.cat((idx, next_id.unsqueeze(0)), dim=1)
            
            result = "".join([IDS[i.item()] for i in idx[0]])
            print(f"  Query: '{q}' -> Result: '{result.replace(q, '').strip()}'")

if __name__ == "__main__":
    run_v2_benchmark()
