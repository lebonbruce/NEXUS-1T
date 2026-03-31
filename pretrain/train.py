"""
NEXUS vs Baseline GPT — 预训练对比实验

使用相同的数据、参数量、超参数训练两个模型，
只有架构不同，对比 loss curve 来验证 NEXUS 核心组件的价值。

数据: WikiText-103 (HuggingFace datasets)
Tokenizer: GPT-2 (50257 vocab)
硬件: RTX 4060 8GB
预计时间: 每个模型 ~2-3 小时 (5000 步)
"""
import os
import sys
import time
import math
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import BaselineGPT, NexusGPT


# ============================================================
# 配置
# ============================================================

class Config:
    """
    训练配置 — Scale-Aware 自动适配版。

    核心参数基于实验数据选择：
      d_model=512, seq_len=1024 → 触发全部组件 (DiffAttn + MLA + TTT + SwiGLU)
      TTT LoRA rank=8 → 实验证明比 full rank 在线学习好 8x
      MoE → d<1024 不启用（50M 规模不需要）

    参数量目标：~25M（NEXUS）vs ~17M（Baseline）
    两者差异来自 TTT 层（这是刻意的——TTT 是额外能力）
    """
    # 模型配置（触发所有 Scale-Aware 组件）
    vocab_size = 50257      # GPT-2 tokenizer
    d_model = 512           # ≥512 → TTT 启用
    n_layers = 6
    n_heads = 8             # head_dim = 512/8/2 = 32 (DiffAttn sub-head)
    seq_len = 1024          # ≥1024 → TTT 启用
    dropout = 0.0

    # Baseline FFN: 4x expansion
    baseline_d_ff = 2048    # 512 * 4
    # NEXUS SwiGLU FFN: 调整到合理范围
    # SwiGLU 有 3 个矩阵(w1,w2,w3)，GELU 有 2 个(fc1,fc2)
    # 为保持 FFN 参数量相近: nexus_d_ff * 3 ≈ baseline_d_ff * 2
    nexus_d_ff = int(512 * 8 / 3)  # ~1365, SwiGLU 标准比例

    # 训练配置
    batch_size = 4          # RTX 4060 8GB（seq_len=1024 需要更小 batch）
    grad_accum_steps = 8    # 有效 batch = 4 * 8 = 32
    lr = 6e-4
    min_lr = 6e-5
    warmup_steps = 200
    max_steps = 5000        # ~2-3 小时
    eval_interval = 250
    eval_steps = 20
    log_interval = 50

    # 数据路径
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ============================================================
# 数据准备
# ============================================================

def prepare_data(config):
    """下载 WikiText-103 并 tokenize。"""
    os.makedirs(config.data_dir, exist_ok=True)

    train_path = os.path.join(config.data_dir, "train.bin")
    val_path = os.path.join(config.data_dir, "val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("  数据已存在，跳过下载")
        train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
        print(f"  训练: {len(train_data):,} tokens | 验证: {len(val_data):,} tokens")
        return train_data, val_data

    print("  准备数据中...")
    print("  [1/3] 加载 tokenizer...")
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    print("  [2/3] 下载 WikiText-103...")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    print("  [3/3] Tokenizing...")
    def tokenize_split(split_name):
        texts = dataset[split_name]["text"]
        all_tokens = []
        for i, text in enumerate(texts):
            if text.strip():
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
            if (i + 1) % 10000 == 0:
                print(f"    {split_name}: {i+1}/{len(texts)} 文档, {len(all_tokens):,} tokens")
        return np.array(all_tokens, dtype=np.uint16)

    train_tokens = tokenize_split("train")
    val_tokens = tokenize_split("validation")

    # 保存为 memmap
    train_mm = np.memmap(train_path, dtype=np.uint16, mode='w+', shape=train_tokens.shape)
    train_mm[:] = train_tokens[:]
    train_mm.flush()

    val_mm = np.memmap(val_path, dtype=np.uint16, mode='w+', shape=val_tokens.shape)
    val_mm[:] = val_tokens[:]
    val_mm.flush()

    print(f"  训练: {len(train_tokens):,} tokens | 验证: {len(val_tokens):,} tokens")
    print(f"  已保存到 {config.data_dir}")

    return train_tokens, val_tokens


def get_batch(data, config, device):
    """从数据中随机采样一个 batch。"""
    ix = torch.randint(0, len(data) - config.seq_len - 1, (config.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+config.seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+config.seq_len+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 训练循环
# ============================================================

def get_lr(step, config):
    """Cosine decay with warmup（标准 LLM 训练学习率调度）。"""
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


@torch.no_grad()
def evaluate(model, val_data, config, device):
    """评估验证集 loss。"""
    model.eval()
    losses = []
    for _ in range(config.eval_steps):
        x, y = get_batch(val_data, config, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train_model(model, model_name, train_data, val_data, config, device):
    """
    训练一个模型，记录 loss 曲线。

    Returns:
        train_losses: list of (step, loss)
        val_losses: list of (step, loss)
    """
    print(f"\n{'='*70}")
    print(f"  训练 {model_name}")
    print(f"  参数量: {model.count_params():,}")
    print(f"  Effective batch: {config.batch_size * config.grad_accum_steps}")
    print(f"  Steps: {config.max_steps} | seq_len: {config.seq_len}")
    print(f"{'='*70}")

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    train_losses = []
    val_losses = []
    start_time = time.time()
    tokens_processed = 0

    for step in range(config.max_steps):
        # 更新学习率
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 梯度累积
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(config.grad_accum_steps):
            x, y = get_batch(train_data, config, device)
            _, loss = model(x, y)
            # 收集 MoE load balancing 辅助损失（如果有 MoE 层）
            if hasattr(model, 'blocks'):
                for block in model.blocks:
                    if hasattr(block, 'ffn') and hasattr(block.ffn, 'aux_loss'):
                        loss = loss + block.ffn.aux_loss
            loss = loss / config.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tokens_processed += config.batch_size * config.seq_len * config.grad_accum_steps

        # 日志
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
            train_losses.append((step, accum_loss))
            print(f"  [{model_name}] step {step:5d}/{config.max_steps} | "
                  f"loss {accum_loss:.4f} | lr {lr:.2e} | "
                  f"{tokens_per_sec:.0f} tok/s | "
                  f"{elapsed:.0f}s elapsed")

        # 评估
        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate(model, val_data, config, device)
            val_losses.append((step, val_loss))
            print(f"  [{model_name}] step {step:5d} | val_loss {val_loss:.4f}")

    # 最终评估
    val_loss = evaluate(model, val_data, config, device)
    val_losses.append((config.max_steps, val_loss))
    total_time = time.time() - start_time

    print(f"\n  [{model_name}] 训练完成!")
    print(f"  最终 val_loss: {val_loss:.4f}")
    print(f"  总时间: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  平均速度: {tokens_processed/total_time:.0f} tokens/s")

    return train_losses, val_losses, val_loss, total_time


# ============================================================
# 文本生成（验证模型是否学到了有意义的东西）
# ============================================================

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, device='cuda'):
    """自回归生成。"""
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # 截断到 seq_len
        idx_cond = tokens[:, -model.seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NEXUS vs Baseline 预训练对比")
    parser.add_argument("--model", type=str, default="both",
                        choices=["baseline", "nexus", "both"],
                        help="训练哪个模型")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = Config()
    config.max_steps = args.max_steps
    config.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  NEXUS vs Baseline GPT 预训练对比实验")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    # 准备数据
    print(f"\n[Step 1] 准备数据...")
    train_data, val_data = prepare_data(config)

    results = {}

    if args.model in ("baseline", "both"):
        # 训练 Baseline GPT
        torch.manual_seed(42)
        baseline = BaselineGPT(
            config.vocab_size, config.d_model, config.n_layers,
            config.n_heads, config.baseline_d_ff, config.seq_len, config.dropout
        )
        b_train, b_val, b_final, b_time = train_model(
            baseline, "Baseline", train_data, val_data, config, device
        )
        results["baseline"] = {
            "train_losses": b_train, "val_losses": b_val,
            "final_val_loss": b_final, "time": b_time,
            "params": baseline.count_params(),
        }
        # 保存模型
        torch.save(baseline.state_dict(),
                   os.path.join(config.data_dir, "baseline_gpt.pt"))

    if args.model in ("nexus", "both"):
        # 训练 NEXUS GPT
        torch.manual_seed(42)
        nexus = NexusGPT(
            config.vocab_size, config.d_model, config.n_layers,
            config.n_heads, config.nexus_d_ff, config.seq_len, config.dropout
        )
        n_train, n_val, n_final, n_time = train_model(
            nexus, "NEXUS", train_data, val_data, config, device
        )
        results["nexus"] = {
            "train_losses": n_train, "val_losses": n_val,
            "final_val_loss": n_final, "time": n_time,
            "params": nexus.count_params(),
        }
        torch.save(nexus.state_dict(),
                   os.path.join(config.data_dir, "nexus_gpt.pt"))

    # 结果对比
    if len(results) == 2:
        print(f"\n\n{'='*70}")
        print(f"  对比结果")
        print(f"{'='*70}")
        b = results["baseline"]
        n = results["nexus"]
        print(f"\n  {'Model':<20s} | {'Params':>10s} | {'Val Loss':>10s} | {'Time':>8s} | {'tok/s':>8s}")
        print(f"  {'-'*65}")
        print(f"  {'Baseline (MHA+GELU)':<20s} | {b['params']:>10,} | {b['final_val_loss']:>10.4f} | "
              f"{b['time']:>7.0f}s | {config.batch_size*config.seq_len*config.grad_accum_steps*config.max_steps/b['time']:>7.0f}")
        print(f"  {'NEXUS (Diff+TTT+SwiG)':<20s} | {n['params']:>10,} | {n['final_val_loss']:>10.4f} | "
              f"{n['time']:>7.0f}s | {config.batch_size*config.seq_len*config.grad_accum_steps*config.max_steps/n['time']:>7.0f}")

        delta = b['final_val_loss'] - n['final_val_loss']
        if delta > 0:
            print(f"\n  ✅ NEXUS val_loss 比 Baseline 低 {delta:.4f} ({delta/b['final_val_loss']*100:.1f}%)")
        else:
            print(f"\n  ❌ NEXUS val_loss 比 Baseline 高 {-delta:.4f} ({-delta/b['final_val_loss']*100:.1f}%)")

        speed_ratio = n['time'] / b['time']
        print(f"  ⏱️  NEXUS 训练速度: {speed_ratio:.2f}x baseline")

        # 生成对比
        try:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            prompt = "The meaning of life is"
            print(f"\n  生成对比 (prompt: '{prompt}'):")
            print(f"\n  Baseline: {generate(baseline, tokenizer, prompt, 50, device=device)}")
            print(f"\n  NEXUS:    {generate(nexus, tokenizer, prompt, 50, device=device)}")
        except Exception as e:
            print(f"\n  生成跳过: {e}")

    # 保存结果
    results_path = os.path.join(config.data_dir, "results.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            "final_val_loss": v["final_val_loss"],
            "time": v["time"],
            "params": v["params"],
            "train_losses": v["train_losses"],
            "val_losses": v["val_losses"],
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  结果已保存到 {results_path}")


if __name__ == "__main__":
    main()
