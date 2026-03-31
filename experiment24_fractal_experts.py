import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Direction Iota: Fractal Experts (分形无限分辨率)
# 核心：专家不再是单层，而是一个可以自我嵌套的无限递归结构
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FractalExpert(nn.Module):
    def __init__(self, d_model, depth=0, max_depth=2):
        super().__init__()
        self.depth = depth
        self.max_depth = max_depth
        
        # 基础计算单元
        self.base_ffn = nn.Linear(d_model, d_model)
        
        # 如果需要更高分辨率，则在内部嵌套子专家
        self.sub_experts = None
        if depth < max_depth:
            self.sub_experts = nn.ModuleList([
                FractalExpert(d_model, depth + 1, max_depth) for _ in range(2)
            ])
            self.router = nn.Linear(d_model, 2)

    def forward(self, x):
        out = self.base_ffn(x)
        if self.sub_experts is not None:
            # 递归路由
            logits = self.router(x)
            weights = F.softmax(logits, dim=-1)
            # 子专家精细化处理
            sub_out = 0
            for i, expert in enumerate(self.sub_experts):
                sub_out += weights[..., i:i+1] * expert(x)
            out = out + sub_out
        return out

def run_iota():
    print(">>> Direction Iota: Starting Fractal Expert Resolution Test...")
    model = FractalExpert(64).to(DEVICE)
    x = torch.randn(1, 16, 64).to(DEVICE)
    out = model(x)
    print(f"  Fractal Depth: {model.max_depth} | Output Shape: {out.shape}")
    print("  [Iota] Fractal hierarchy initialized. Resolution scale is now multi-level.")

if __name__ == "__main__":
    run_iota()
