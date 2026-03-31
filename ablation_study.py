"""
消融实验：分别验证SCE三个核心机制的独立贡献。

变体：
1. StdAttn + Growth + KD   (去掉Delta，用标准Attention)
2. StdAttn + Growth - KD   (去掉Delta和KD，纯结构隔离)
3. StdAttn - Growth + KD   (不隔离，只有KD正则化)
4. Progressive             (对照组：完全隔离)
5. Replay                  (对照组：回放)

关键问题：哪个机制贡献最大？能否综合？
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from sce.config import ExperimentConfig
from sce.tasks import generate_task_data, TASK_NAMES
from sce.evaluation import compute_accuracy, compute_cl_metrics
from sce.models.base import CLModel, TransformerBlock
from sce.models.components.surprise_growth import SurpriseFFN
from sce.models.components.consolidation import compute_kd_loss


# ============================================================
# 消融变体1: StdAttn + Growth + KD (最完整，但用标准Attention)
# ============================================================
class AblationBlock_Full(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads,
            batch_first=True, dropout=config.dropout
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = SurpriseFFN(config.d_model, config.d_ff)

    def forward(self, x, task_id=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.ln2(x), task_id=task_id)
        return x


class Ablation_GrowthKD(CLModel):
    """StdAttn + Expert Growth + KD（去掉Delta，保留其余两个机制）"""

    def __init__(self, config):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.blocks = nn.ModuleList([AblationBlock_Full(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        self._kd_alpha = config.kd_alpha
        self._has_old = False

    def forward(self, x, task_id):
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)
        for block in self.blocks:
            h = block(h, task_id=task_id)
        return self.head(self.ln_f(h))

    def forward_with_kd(self, x, task_id):
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)
        kd_loss = torch.tensor(0.0, device=device)
        kd_count = 0
        for block in self.blocks:
            h_ln = block.ln1(h)
            h = h + block.attn(h_ln, h_ln, h_ln, need_weights=False)[0]
            h_ffn_in = block.ln2(h)
            ffn_out = block.ffn(h_ffn_in, task_id=task_id)
            if self._has_old and self.training:
                old_out = block.ffn.get_old_expert_output(h_ffn_in)
                if old_out is not None:
                    kd_loss = kd_loss + F.mse_loss(ffn_out, old_out)
                    kd_count += 1
            h = h + ffn_out
        logits = self.head(self.ln_f(h))
        if kd_count > 0:
            kd_loss = self._kd_alpha * kd_loss / kd_count
        return logits, kd_loss

    def on_task_start(self, task_id):
        if task_id > 0:
            device = next(self.parameters()).device
            for block in self.blocks:
                block.ffn.grow(device)
            self._has_old = True
        for block in self.blocks:
            block.ffn.register_task(task_id)

    def on_task_end(self, task_id, train_x, train_y):
        for block in self.blocks:
            block.ffn.register_task(task_id)


# ============================================================
# 消融变体2: StdAttn + Growth - KD (纯结构隔离，无蒸馏)
# ============================================================
class Ablation_GrowthOnly(Ablation_GrowthKD):
    """StdAttn + Expert Growth，无KD"""

    def forward_with_kd(self, x, task_id):
        logits = self.forward(x, task_id)
        return logits, torch.tensor(0.0, device=x.device)


# ============================================================
# 消融变体3: StdAttn + KD - Growth (不隔离Expert，只用KD正则化)
# ============================================================
class Ablation_KDOnly(CLModel):
    """StdAttn + KD正则化，不冻结/不隔离Expert"""

    def __init__(self, config):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        self._old_params = None
        self._kd_alpha = config.kd_alpha

    def forward(self, x, task_id):
        B, T = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        task = torch.full((B,), task_id, dtype=torch.long, device=device)
        h = self.token_emb(x) + self.pos_emb(pos) + self.task_emb(task).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

    def forward_with_kd(self, x, task_id):
        logits = self.forward(x, task_id)
        return logits, torch.tensor(0.0, device=x.device)

    def compute_extra_loss(self):
        # 简化版KD：权重不要偏离太远(类似EWC但无Fisher)
        if self._old_params is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for (name, param) in self.named_parameters():
            if name in self._old_params:
                loss = loss + ((param - self._old_params[name]) ** 2).sum()
        return self._kd_alpha * loss

    def on_task_end(self, task_id, train_x, train_y):
        self._old_params = {n: p.data.clone() for n, p in self.named_parameters()}


# ============================================================
# 统一训练/评估循环
# ============================================================
def run_ablation(config, model_class, name, device, use_kd_forward=False):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    test_data = []
    for tid in range(config.num_tasks):
        tx, ty = generate_task_data(tid, config.test_samples, config.vocab_size, config.seq_len)
        test_data.append((tx, ty))

    model = model_class(config).to(device)
    acc_matrix = np.zeros((config.num_tasks, config.num_tasks))

    t0 = time.time()
    for task_id in range(config.num_tasks):
        train_x, train_y = generate_task_data(task_id, config.train_samples, config.vocab_size, config.seq_len)
        model.on_task_start(task_id)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=config.lr)
        model.train()

        for step in range(config.steps_per_task):
            idx = torch.randint(0, len(train_x), (config.batch_size,))
            bx, by = train_x[idx].to(device), train_y[idx].to(device)

            if use_kd_forward and hasattr(model, 'forward_with_kd'):
                logits, kd_loss = model.forward_with_kd(bx, task_id)
                task_loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), by.reshape(-1))
                loss = task_loss + kd_loss
            else:
                logits = model(bx, task_id)
                task_loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), by.reshape(-1))
                loss = task_loss + model.compute_extra_loss()

            # Replay if available
            replay = model.get_replay_data(config.batch_size // 2)
            if replay is not None:
                rx, ry, rt = replay
                r_logits = model(rx, rt)
                loss = loss + F.cross_entropy(r_logits.reshape(-1, config.vocab_size), ry.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.on_task_end(task_id, train_x, train_y)

        for eid in range(config.num_tasks):
            tx, ty = test_data[eid]
            acc_matrix[eid, task_id] = compute_accuracy(model, tx, ty, eid, device)

    duration = time.time() - t0
    metrics = compute_cl_metrics(acc_matrix)
    total_p = model.count_params(False)

    # 打印per-task结果
    print(f"  Time: {duration:.0f}s | Params: {total_p/1e3:.0f}K")
    for i in range(config.num_tasks):
        print(f"    {TASK_NAMES[i]:15s}: {acc_matrix[i,-1]*100:.1f}%")
    print(f"  AA={metrics['AA']*100:.1f}%  BWT={metrics['BWT']*100:+.1f}%")

    return {"name": name, "aa": metrics["AA"], "bwt": metrics["BWT"],
            "per_task": acc_matrix[:, -1], "params": total_p, "time": duration}


def main():
    config = ExperimentConfig()
    config.num_runs = 1  # 消融只跑1次，快速验证
    device = torch.device(config.device)
    torch.manual_seed(42)
    np.random.seed(42)

    from sce.models.naive import NaiveTransformer
    from sce.models.replay import ReplayTransformer
    from sce.models.progressive import ProgressiveNet

    results = []

    # 对照组
    torch.manual_seed(42); np.random.seed(42)
    results.append(run_ablation(config, NaiveTransformer, "Naive (baseline)", device))

    torch.manual_seed(42); np.random.seed(42)
    results.append(run_ablation(config, ReplayTransformer, "Replay (baseline)", device))

    torch.manual_seed(42); np.random.seed(42)
    results.append(run_ablation(config, ProgressiveNet, "Progressive (baseline)", device))

    # 消融变体
    torch.manual_seed(42); np.random.seed(42)
    results.append(run_ablation(config, Ablation_KDOnly, "KD Only (no isolation)", device))

    torch.manual_seed(42); np.random.seed(42)
    results.append(run_ablation(config, Ablation_GrowthOnly, "Growth Only (no KD)", device, use_kd_forward=True))

    torch.manual_seed(42); np.random.seed(42)
    results.append(run_ablation(config, Ablation_GrowthKD, "Growth + KD (full)", device, use_kd_forward=True))

    # 汇总表
    print("\n\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    header = f"{'Method':25s} | {'AA':>6s} | {'BWT':>7s} | {'Params':>7s} | {'Time':>5s}"
    print(header)
    print("-" * 70)
    for r in results:
        aa = f"{r['aa']*100:.1f}%"
        bwt = f"{r['bwt']*100:+.1f}%"
        params = f"{r['params']/1e3:.0f}K"
        t = f"{r['time']:.0f}s"
        print(f"{r['name']:25s} | {aa:>6s} | {bwt:>7s} | {params:>7s} | {t:>5s}")

    print("-" * 70)
    print("\nPer-Task Accuracy (after all 5 tasks):")
    print(f"{'Method':25s} |", end="")
    for tn in TASK_NAMES:
        print(f" {tn[:8]:>8s}", end="")
    print()
    print("-" * 70)
    for r in results:
        print(f"{r['name']:25s} |", end="")
        for v in r['per_task']:
            print(f" {v*100:7.1f}%", end="")
        print()

    # 分析
    print("\n" + "=" * 70)
    print("MECHANISM CONTRIBUTION ANALYSIS")
    print("=" * 70)
    naive_aa = results[0]["aa"]
    kd_aa = results[3]["aa"]
    growth_aa = results[4]["aa"]
    full_aa = results[5]["aa"]
    prog_aa = results[2]["aa"]

    print(f"Naive baseline:              {naive_aa*100:.1f}%")
    print(f"+ KD Only (no isolation):    {kd_aa*100:.1f}% (Δ = {(kd_aa-naive_aa)*100:+.1f}pp)")
    print(f"+ Growth Only (isolation):   {growth_aa*100:.1f}% (Δ = {(growth_aa-naive_aa)*100:+.1f}pp)")
    print(f"+ Growth + KD (both):        {full_aa*100:.1f}% (Δ = {(full_aa-naive_aa)*100:+.1f}pp)")
    print(f"Progressive (upper bound):   {prog_aa*100:.1f}% (Δ = {(prog_aa-naive_aa)*100:+.1f}pp)")
    print()
    print("Key insight: Isolation contributes "
          f"{(growth_aa-naive_aa)/(prog_aa-naive_aa)*100:.0f}% of Progressive's gain")


if __name__ == "__main__":
    main()
