"""
长序列CL Benchmark: 让TTT的test-time learning真正发挥作用

关键改进：
  seq_len=64（之前是16），给TTT足够的token来在forward中学习模式
  任务都有长距离依赖（不是逐token独立的）
  持久化W1的更新使用完整序列梯度（不是只用最后一个token）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


# ============================================================
# 长距离依赖任务（seq_len=64）
# ============================================================
VOCAB = 32
SEQ_LEN = 64
TASK_NAMES = ["PrefixSum", "Reverse", "Echo-8", "RunMax"]

def gen_task(tid, n):
    """生成长序列任务数据。所有任务都需要记住序列历史。"""
    x = torch.randint(1, VOCAB, (n, SEQ_LEN))
    if tid == 0:  # PrefixSum: y_i = cumsum(x) mod VOCAB
        y = x.cumsum(dim=1) % VOCAB
    elif tid == 1:  # Reverse: y_i = x_{L-1-i}
        y = x.flip(1)
    elif tid == 2:  # Echo-8: y_i = x_{i-8} (延迟回声)
        y = torch.zeros_like(x)
        y[:, 8:] = x[:, :-8]
        y[:, :8] = x[:, :8]  # 前8个位置复制自身
    elif tid == 3:  # RunMax: y_i = max(x_0..x_i)
        y = x.cummax(dim=1).values
    return x, y


# ============================================================
# 模型定义
# ============================================================
class BaseModel(nn.Module):
    def __init__(self, d=128, nh=4, nl=4, ff=256):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQ_LEN, d)
        self.task_emb = nn.Embedding(len(TASK_NAMES), d)

    def embed(self, x, tid):
        B, T = x.size()
        dev = x.device
        return self.tok(x) + self.pos(torch.arange(T, device=dev)) + \
               self.task_emb(torch.full((B,), tid, dtype=torch.long, device=dev)).unsqueeze(1)

    def count_p(self):
        return sum(p.numel() for p in self.parameters())

    def extra_loss(self):
        return 0.0

    def get_replay(self, bs):
        return None

    def after_task(self, tid, tx, ty):
        pass


class NaiveModel(BaseModel):
    """标准Transformer baseline。"""
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(128, 4, 256, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 4)
        self.head = nn.Linear(128, VOCAB)

    def forward(self, x, tid):
        return self.head(self.enc(self.embed(x, tid)))


class ReplayModel(NaiveModel):
    """Naive + 经验回放。"""
    def __init__(self):
        super().__init__()
        self._bx, self._by, self._bt = [], [], []

    def after_task(self, tid, tx, ty):
        idx = torch.randperm(len(tx))[:300]
        self._bx.append(tx[idx].cpu()); self._by.append(ty[idx].cpu()); self._bt.append(tid)

    def get_replay(self, bs):
        if not self._bx: return None
        bi = torch.randint(0, len(self._bx), (1,)).item()
        si = torch.randint(0, len(self._bx[bi]), (min(bs, len(self._bx[bi])),))
        dev = next(self.parameters()).device
        return self._bx[bi][si].to(dev), self._by[bi][si].to(dev), self._bt[bi]


# ============================================================
# TTT Layer（简化版，核心逻辑来自官方ttt.py）
# ============================================================
class TTTLayer(nn.Module):
    def __init__(self, d=128, nh=4, persistent=False, ttt_lr=1.0):
        super().__init__()
        self.d, self.nh, self.hd = d, nh, d // nh
        self.persistent = persistent
        self.ttt_base_lr = ttt_lr

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.W1_init = nn.Parameter(torch.normal(0, 0.02, size=(nh, d//nh, d//nh)))
        self.b1_init = nn.Parameter(torch.zeros(nh, 1, d//nh))

        self.ttt_lr_w = nn.Parameter(torch.normal(0, 0.02, size=(nh, d, 1)))
        self.ttt_lr_b = nn.Parameter(torch.zeros(nh, 1))
        self.ln_w = nn.Parameter(torch.ones(nh, d//nh))
        self.ln_b = nn.Parameter(torch.zeros(nh, d//nh))
        self.post_norm = nn.LayerNorm(d)

        # 持久化状态
        if persistent:
            self.register_buffer('W1_s', torch.normal(0, 0.02, size=(nh, d//nh, d//nh)))
            self.register_buffer('b1_s', torch.zeros(nh, 1, d//nh))
            self._inited = False

    def reset(self):
        if self.persistent:
            with torch.no_grad():
                self.W1_s.copy_(self.W1_init.data)
                self.b1_s.copy_(self.b1_init.data)
            self._inited = True

    def forward(self, h):
        B, L, _ = h.shape
        nh, hd = self.nh, self.hd

        XQ = self.q_proj(h).view(B,L,nh,hd).permute(0,2,1,3)
        XK = self.k_proj(h).view(B,L,nh,hd).permute(0,2,1,3)
        XV = self.v_proj(h).view(B,L,nh,hd).permute(0,2,1,3)

        # 使用持久化或初始W1
        if self.persistent and self._inited:
            W1 = self.W1_s.unsqueeze(0).expand(B,-1,-1,-1)
            b1 = self.b1_s.unsqueeze(0).expand(B,-1,-1,-1)
        else:
            W1 = self.W1_init.unsqueeze(0).expand(B,-1,-1,-1)
            b1 = self.b1_init.unsqueeze(0).expand(B,-1,-1,-1)

        # 数据依赖学习率
        eta = torch.einsum('bld,hdo->bhlo', h, self.ttt_lr_w)
        eta = torch.sigmoid(eta + self.ttt_lr_b.view(1,nh,1,1)) * self.ttt_base_lr / hd

        # 自监督：从K预测V-K
        Z1 = XK @ W1 + b1
        target = XV - XK
        # LayerNorm + L2的梯度
        w = self.ln_w.view(nh,1,hd); b = self.ln_b.view(nh,1,hd)
        mu = Z1.mean(-1, keepdim=True)
        std = (Z1.var(dim=-1, keepdim=True, correction=0) + 1e-6).sqrt()
        xh = (Z1 - mu) / std
        y = w * xh + b
        go = (y - target) * w
        grad = (1.0/hd) * (hd*go - go.sum(-1,True) - xh*(go*xh).sum(-1,True)) / std

        # Dual form
        A = torch.tril(XQ @ XK.transpose(-2,-1))
        et = torch.tril(eta.expand(-1,-1,-1,L))
        b1_bar = b1 - et @ grad
        Z_bar = XQ @ W1 - (eta * A) @ grad + b1_bar
        # LayerNorm输出
        mu2 = Z_bar.mean(dim=-1, keepdim=True); std2 = (Z_bar.var(dim=-1, keepdim=True, correction=0)+1e-6).sqrt()
        Z_bar = w * (Z_bar-mu2)/std2 + b
        out = XQ + Z_bar

        # 持久化更新：用完整序列的梯度更新W1_state
        if self.persistent and self.training:
            with torch.no_grad():
                dW = (eta * XK).transpose(-2,-1) @ grad  # [B,nh,hd,hd]
                db = (eta * grad).sum(2, keepdim=True)     # [B,nh,1,hd]
                # 对batch取平均，EMA更新
                self.W1_s -= 0.1 * dW.mean(0)
                self.b1_s -= 0.1 * db.mean(0)

        out = out.permute(0,2,1,3).reshape(B,L,self.d)
        return self.o_proj(self.post_norm(out))


class TTTBlock(nn.Module):
    def __init__(self, persistent=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(128)
        self.ttt = TTTLayer(persistent=persistent)
        self.ln2 = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128,256), nn.GELU(), nn.Linear(256,128))

    def forward(self, x):
        x = x + self.ttt(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TTTModel(BaseModel):
    def __init__(self, persistent=False):
        super().__init__()
        self._persistent = persistent
        self.blocks = nn.ModuleList([TTTBlock(persistent) for _ in range(4)])
        self.ln_f = nn.LayerNorm(128)
        self.head = nn.Linear(128, VOCAB)

    def forward(self, x, tid):
        h = self.embed(x, tid)
        for b in self.blocks: h = b(h)
        return self.head(self.ln_f(h))

    def after_task(self, tid, tx, ty):
        pass


class TTTReplayModel(TTTModel):
    def __init__(self, persistent=False):
        super().__init__(persistent)
        self._bx, self._by, self._bt = [], [], []

    def after_task(self, tid, tx, ty):
        idx = torch.randperm(len(tx))[:300]
        self._bx.append(tx[idx].cpu()); self._by.append(ty[idx].cpu()); self._bt.append(tid)

    def get_replay(self, bs):
        if not self._bx: return None
        bi = torch.randint(0, len(self._bx), (1,)).item()
        si = torch.randint(0, len(self._bx[bi]), (min(bs, len(self._bx[bi])),))
        dev = next(self.parameters()).device
        return self._bx[bi][si].to(dev), self._by[bi][si].to(dev), self._bt[bi]


# ============================================================
# 运行
# ============================================================
def run(model, name, device):
    print(f"\n  [{name}]", end="", flush=True)
    NT = len(TASK_NAMES)
    test = [gen_task(t, 500) for t in range(NT)]
    model = model.to(device)
    acc = np.zeros((NT, NT))
    t0 = time.time()

    for tid in range(NT):
        tx, ty = gen_task(tid, 2000)
        # 初始化持久化状态
        if tid == 0:
            for b in (model.blocks if hasattr(model, 'blocks') else []):
                if hasattr(b, 'ttt') and hasattr(b.ttt, 'reset'):
                    b.ttt.reset()

        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        model.train()
        for step in range(400):
            idx = torch.randint(0, len(tx), (64,))
            bx, by = tx[idx].to(device), ty[idx].to(device)
            logits = model(bx, tid)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
            replay = model.get_replay(32)
            if replay:
                rx, ry, rt = replay
                loss = loss + F.cross_entropy(model(rx, rt).reshape(-1, VOCAB), ry.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 99:
                print(".", end="", flush=True)

        model.after_task(tid, tx, ty)
        model.eval()
        with torch.no_grad():
            for eid in range(NT):
                ex, ey = test[eid]
                correct = 0; total = 0
                for i in range(0, len(ex), 100):
                    bx = ex[i:i+100].to(device)
                    by = ey[i:i+100].to(device)
                    pred = model(bx, eid).argmax(-1)
                    correct += (pred == by).sum().item()
                    total += by.numel()
                acc[eid, tid] = correct / total
        print(f" T{tid}:{acc[tid,tid]*100:.0f}%", end="", flush=True)

    dur = time.time() - t0
    # CL指标
    aa = acc[:, -1].mean()
    bwt = 0
    for i in range(NT - 1):
        bwt += acc[i, -1] - acc[i, i]
    bwt /= (NT - 1)
    p = model.count_p()
    print(f" | AA={aa*100:.1f}% BWT={bwt*100:+.1f}% T={dur:.0f}s P={p/1e3:.0f}K")
    return {"name": name, "aa": aa, "bwt": bwt, "per": acc[:,-1], "p": p, "t": dur}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  LONG-SEQUENCE CL BENCHMARK (seq_len=64)")
    print(f"  Tasks: {TASK_NAMES}")
    print(f"  Device: {device}")
    print("=" * 65)

    methods = [
        (lambda: NaiveModel(),                    "Naive-Attn"),
        (lambda: ReplayModel(),                   "Replay-Attn"),
        (lambda: TTTModel(persistent=False),      "TTT"),
        (lambda: TTTReplayModel(persistent=False), "TTT+Replay"),
        (lambda: TTTModel(persistent=True),       "TTT-Persist"),
        (lambda: TTTReplayModel(persistent=True),  "TTT-Persist+Replay"),
    ]
    results = []
    for factory, name in methods:
        torch.manual_seed(42); np.random.seed(42)
        results.append(run(factory(), name, device))

    print("\n\n" + "=" * 70)
    print(f"{'Method':22s} | {'AA':>6s} | {'BWT':>7s} | {'Params':>7s} | {'Time':>5s}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:22s} | {r['aa']*100:5.1f}% | {r['bwt']*100:+5.1f}% | "
              f"{r['p']/1e3:5.0f}K | {r['t']:4.0f}s")
    print("-" * 70)
    print(f"\n{'Per-Task':22s} |", end="")
    for tn in TASK_NAMES: print(f" {tn:>10s}", end="")
    print()
    for r in results:
        print(f"{r['name']:22s} |", end="")
        for v in r['per']: print(f" {v*100:9.1f}%", end="")
        print()

    # 关键对比
    print("\n" + "=" * 70)
    print("KEY COMPARISONS (long-seq vs short-seq)")
    print("=" * 70)
    naive_aa = results[0]['aa']
    replay_aa = results[1]['aa']
    ttt_aa = results[2]['aa']
    tttp_aa = results[4]['aa']
    print(f"  TTT vs Naive:              {ttt_aa*100:.1f}% vs {naive_aa*100:.1f}% (diff: {(ttt_aa-naive_aa)*100:+.1f}%)")
    print(f"  TTT-Persist vs TTT:        {tttp_aa*100:.1f}% vs {ttt_aa*100:.1f}% (diff: {(tttp_aa-ttt_aa)*100:+.1f}%)")
    print(f"  TTT-Persist+R vs Replay:   {results[5]['aa']*100:.1f}% vs {replay_aa*100:.1f}% (diff: {(results[5]['aa']-replay_aa)*100:+.1f}%)")


if __name__ == "__main__":
    main()
