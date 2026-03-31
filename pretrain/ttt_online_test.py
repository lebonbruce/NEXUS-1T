"""
TTT 在线适应能力专项测试 — Test-Time Training 的真正舞台

核心设计理念：
  TTT 的核心不是"训练更快"，而是"推理时即时学习"。
  本测试不比训练速率，而是比 forward pass 中的在线适应能力。

两个维度的测试：

  TEST 1: Per-Position Loss 衰减
    原理：TTT 在 forward 中通过 cumsum 累积梯度更新 W，
    序列越到后面 W 越适应当前输入 → loss 应该随位置下降。
    Baseline 没有这种机制 → loss 不应该有"学习"趋势。

  TEST 2: Induction Pattern 识别
    原理：在序列中重复出现 [A, B] 对。
    TTT 应该在第一次看到 [A, B] 后学会 "A → B" 的关联，
    下次看到 A 时能更好地预测 B。
    这是 TTT "用结构换算力" 的直接证据。

注意：不需要预训练！随机权重即可测试在线学习能力。
TTT 层在 forward pass 中的自监督更新是独立于预训练的。
"""

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain"))
from models import BaselineGPT, NexusGPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TEST 1: Per-Position Loss 衰减（TTT 在线学习的直接证据）
# ============================================================

def test_per_position_loss():
    """
    测量 loss 随 token 位置的变化。

    如果 TTT 在 forward pass 中真的在"学习"，那么：
      - 序列后半段的 loss 应该比前半段低
      - 这个"下降趋势"应该比 Baseline 更明显

    方法：
      1. 生成有明确重复模式的合成序列
      2. 计算每 64 个 token 区间的平均 loss
      3. 对比 Baseline 和 NEXUS（全组件开启）
    """
    print("=" * 70)
    print("  TEST 1: Per-Position Loss 衰减（在线学习直接证据）")
    print("=" * 70)

    V = 256  # 小词表（合成数据不需要大词表）
    D = 512
    L = 4
    H = 8
    SEQ = 1024
    BATCH = 4

    BL_DFF = D * 4  # 2048
    NX_DFF = int(D * 4 * 2 / 3)  # ~1365

    # 生成有重复模式的合成序列
    # 策略：创建一个 "主题库"（64 个短模式），然后不断重复采样
    # TTT 应该能学到这些模式的统计规律
    torch.manual_seed(42)
    np.random.seed(42)

    patterns = []
    for _ in range(64):
        pattern_len = np.random.randint(4, 16)
        patterns.append(torch.randint(1, V, (pattern_len,)))

    def generate_patterned_sequence(batch_size, seq_len):
        """生成有重复模式的序列。"""
        seqs = []
        for _ in range(batch_size):
            tokens = []
            while len(tokens) < seq_len + 1:
                # 随机选择一个模式并追加
                p = patterns[np.random.randint(0, len(patterns))]
                tokens.extend(p.tolist())
            seqs.append(torch.tensor(tokens[:seq_len + 1]))
        batch = torch.stack(seqs)
        return batch[:, :seq_len].to(DEVICE), batch[:, 1:seq_len + 1].to(DEVICE)

    results = {}
    for name, model_cls, dff in [
        ("Baseline", BaselineGPT, BL_DFF),
        ("NEXUS (全组件)", NexusGPT, NX_DFF),
    ]:
        torch.manual_seed(42)
        model = model_cls(V, D, L, H, dff, SEQ).to(DEVICE).eval()
        params = model.count_params()

        # 多次采样取平均
        n_trials = 8
        # 将序列分成 16 个区间，每个 64 token
        n_bins = SEQ // 64
        bin_losses = np.zeros(n_bins)

        with torch.no_grad():
            for _ in range(n_trials):
                x, y = generate_patterned_sequence(BATCH, SEQ)
                logits, _ = model(x)

                # 计算 per-position loss
                for b in range(n_bins):
                    start = b * 64
                    end = (b + 1) * 64
                    bin_logits = logits[:, start:end, :].reshape(-1, V)
                    bin_targets = y[:, start:end].reshape(-1)
                    bin_loss = F.cross_entropy(bin_logits, bin_targets).item()
                    bin_losses[b] += bin_loss / n_trials

        # 计算趋势：前半 vs 后半
        first_half = np.mean(bin_losses[:n_bins // 2])
        second_half = np.mean(bin_losses[n_bins // 2:])
        improvement = first_half - second_half  # 正值 = 后半段更好

        results[name] = {
            "bin_losses": bin_losses.tolist(),
            "first_half": first_half,
            "second_half": second_half,
            "improvement": improvement,
            "params": params,
        }

        print(f"\n  [{name}] params={params / 1e6:.1f}M")
        print(f"  Position |", end="")
        for b in range(0, n_bins, 2):
            print(f" {b * 64:>4d}-{(b + 1) * 64:>4d}", end="")
        print()
        print(f"  Loss     |", end="")
        for b in range(0, n_bins, 2):
            print(f"  {bin_losses[b]:>7.4f}", end="")
        print()

        print(f"\n  前半段平均 loss: {first_half:.4f}")
        print(f"  后半段平均 loss: {second_half:.4f}")
        print(f"  改善量（后-前）: {improvement:+.4f}")
        if improvement > 0.01:
            print(f"  ✅ 检测到在线学习信号！后半段比前半段好 {improvement:.4f}")
        else:
            print(f"  ❌ 未检测到显著在线学习信号")

        del model
        torch.cuda.empty_cache()

    # 对比
    bl_imp = results["Baseline"]["improvement"]
    nx_imp = results["NEXUS (全组件)"]["improvement"]
    diff = nx_imp - bl_imp

    print(f"\n  --- 在线学习能力对比 ---")
    print(f"  Baseline 在线改善量: {bl_imp:+.4f}")
    print(f"  NEXUS    在线改善量: {nx_imp:+.4f}")
    print(f"  NEXUS 额外贡献:     {diff:+.4f}")
    if diff > 0.01:
        print(f"  ✅ TTT 确实提供了额外的在线适应能力！(+{diff:.4f})")
    elif diff > -0.01:
        print(f"  ≈ TTT 在线适应与 Baseline 持平")
    else:
        print(f"  ⚠️ TTT 表现不如 Baseline 的自然注意力适应")

    return results


# ============================================================
# TEST 2: Induction Pattern（关联记忆测试）
# ============================================================

def test_induction_pattern():
    """
    Induction Head 测试：[..., A, B, ..., A, ?] → 模型应预测 B。

    这是 Transformer 最基本的 in-context learning 能力之一。
    TTT 的在线权重更新应该让它更快地识别 A→B 关联。

    设计：
      1. 创建 N 个唯一的 (A, B) 关联对
      2. 在序列前半段呈现这些对
      3. 在序列后半段，给出 A，看模型是否预测 B
      4. 测量"第一次见 vs 第二次见 vs 第三次见"的准确率
    """
    print("\n" + "=" * 70)
    print("  TEST 2: Induction Pattern 识别（A→B 关联记忆）")
    print("=" * 70)

    V = 256
    D = 512
    L = 4
    H = 8
    SEQ = 1024
    BATCH = 8

    BL_DFF = D * 4
    NX_DFF = int(D * 4 * 2 / 3)

    torch.manual_seed(42)
    np.random.seed(42)

    def generate_induction_sequence(batch_size, seq_len, n_pairs=16):
        """
        生成 induction 测试序列。

        结构：
          [填充] [A1, B1] [填充] [A2, B2] [填充] ... [A1, ?] [填充] [A2, ?] ...

        每个 (A, B) 对在序列中出现多次。
        我们会标记每次出现的位置以便后续分析。
        """
        all_x = []
        all_y = []
        # 记录每个 (A, B) 对在不同出现次数时的位置
        # occurrence_positions[occ_count] = list of (batch_idx, position)
        pair_positions = {1: [], 2: [], 3: [], 4: []}

        for bi in range(batch_size):
            # 生成 n_pairs 个唯一的 (A, B) 对
            keys = torch.randperm(V)[:n_pairs * 2].reshape(n_pairs, 2)
            As = keys[:, 0]
            Bs = keys[:, 1]

            # 构建序列：每个对重复出现多次，中间填充随机 token
            tokens = []
            targets = []
            pair_count = np.zeros(n_pairs, dtype=int)  # 每对出现次数

            # 填充到 seq_len
            while len(tokens) < seq_len:
                # 随机决定：放一个 pair 还是放随机 token
                if np.random.random() < 0.3 and len(tokens) < seq_len - 2:
                    # 放一个 pair
                    pi = np.random.randint(0, n_pairs)
                    a, b = As[pi].item(), Bs[pi].item()
                    pair_count[pi] += 1
                    occ = int(pair_count[pi])

                    tokens.append(a)
                    targets.append(b)  # 这里 target 是正确的 B

                    pos = len(tokens) - 1
                    if occ <= 4:
                        pair_positions[occ].append((bi, pos))

                    tokens.append(b)
                    targets.append(np.random.randint(1, V))  # B 后面的 target 随便
                else:
                    # 随机填充
                    t = np.random.randint(1, V)
                    tokens.append(t)
                    targets.append(np.random.randint(1, V))

            all_x.append(torch.tensor(tokens[:seq_len]))
            all_y.append(torch.tensor(targets[:seq_len]))

        x = torch.stack(all_x).to(DEVICE)
        y = torch.stack(all_y).to(DEVICE)
        return x, y, pair_positions

    results = {}
    for name, model_cls, dff in [
        ("Baseline", BaselineGPT, BL_DFF),
        ("NEXUS (全组件)", NexusGPT, NX_DFF),
    ]:
        torch.manual_seed(42)
        model = model_cls(V, D, L, H, dff, SEQ).to(DEVICE).eval()

        # 多次试验
        n_trials = 5
        occ_losses = {1: [], 2: [], 3: [], 4: []}
        occ_accs = {1: [], 2: [], 3: [], 4: []}

        with torch.no_grad():
            for trial in range(n_trials):
                np.random.seed(trial * 100)
                x, y, positions = generate_induction_sequence(BATCH, SEQ)
                logits, _ = model(x)
                probs = F.softmax(logits, dim=-1)

                for occ in [1, 2, 3, 4]:
                    if not positions[occ]:
                        continue
                    losses = []
                    correct = 0
                    total = 0
                    for bi, pos in positions[occ]:
                        if pos < SEQ:
                            target = y[bi, pos].item()
                            loss = F.cross_entropy(
                                logits[bi, pos].unsqueeze(0),
                                y[bi, pos].unsqueeze(0)
                            ).item()
                            losses.append(loss)
                            pred = logits[bi, pos].argmax().item()
                            if pred == target:
                                correct += 1
                            total += 1
                    if losses:
                        occ_losses[occ].append(np.mean(losses))
                    if total > 0:
                        occ_accs[occ].append(correct / total)

        print(f"\n  [{name}]")
        print(f"  {'Occurrence':>12s} | {'Avg Loss':>10s} | {'Accuracy':>10s} | {'Trend'}")
        print(f"  {'-' * 55}")

        prev_loss = None
        for occ in [1, 2, 3, 4]:
            if occ_losses[occ]:
                avg_loss = np.mean(occ_losses[occ])
                avg_acc = np.mean(occ_accs[occ]) * 100 if occ_accs[occ] else 0
                trend = ""
                if prev_loss is not None:
                    diff = avg_loss - prev_loss
                    trend = f"{'↓' if diff < -0.01 else '↑' if diff > 0.01 else '='} {diff:+.3f}"
                prev_loss = avg_loss
                print(f"  {'1st见' if occ==1 else f'{occ}nd见' if occ==2 else f'{occ}rd见' if occ==3 else f'{occ}th见':>12s}"
                      f" | {avg_loss:>10.4f} | {avg_acc:>9.1f}% | {trend}")

        results[name] = {
            "losses": {k: np.mean(v) if v else None for k, v in occ_losses.items()},
            "accs": {k: np.mean(v) * 100 if v else None for k, v in occ_accs.items()},
        }

        del model
        torch.cuda.empty_cache()

    # 对比：第 1 次见 vs 第 3 次见的 loss 改善量
    print(f"\n  --- 关联学习能力对比 ---")
    for name in results:
        losses = results[name]["losses"]
        if losses[1] is not None and losses[3] is not None:
            improve = losses[1] - losses[3]
            print(f"  {name}: 1st→3rd loss 改善 = {improve:+.4f}")

    bl_losses = results["Baseline"]["losses"]
    nx_losses = results["NEXUS (全组件)"]["losses"]
    if bl_losses[1] and bl_losses[3] and nx_losses[1] and nx_losses[3]:
        bl_improve = bl_losses[1] - bl_losses[3]
        nx_improve = nx_losses[1] - nx_losses[3]
        diff = nx_improve - bl_improve
        print(f"\n  TTT 额外关联学习增益: {diff:+.4f}")
        if diff > 0.05:
            print(f"  ✅ TTT 在关联学习上展现了明显优势！")
            print(f"     每次重复看到 A→B 模式时，TTT 的适应速度更快")
        elif diff > 0:
            print(f"  ✅ TTT 有轻微优势")
        else:
            print(f"  ⚠️ 在随机权重下 TTT 未展现关联学习优势")
            print(f"     这可能是因为 TTT 的自监督目标需要预训练基础才能发挥")

    return results


# ============================================================
# TEST 3: 序列内 Loss 下降斜率（最直接的在线学习指标）
# ============================================================

def test_loss_slope():
    """
    测量序列内 loss 的"下降斜率"。

    这是最直接的 TTT 在线学习指标：
      斜率 < 0: 模型在 forward 过程中越来越好（在线学习有效）
      斜率 ≈ 0: 模型对每个位置一视同仁（无在线学习）
      斜率 > 0: 模型在长序列中退化（注意力稀释等）

    用 WikiText 真实数据来测试！
    """
    print("\n" + "=" * 70)
    print("  TEST 3: 序列内 Loss 下降斜率（真实数据）")
    print("=" * 70)

    # 查找数据
    data_dir = None
    for base in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain", "data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        os.path.join("pretrain", "data"),
    ]:
        if os.path.exists(os.path.join(base, "val.bin")):
            data_dir = base
            break

    if data_dir is None:
        print("  ⚠️ 验证数据未找到，跳过")
        return None

    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    V = 50257  # WikiText 词表
    D = 512
    L = 4
    H = 8
    SEQ = 1024
    BATCH = 2

    BL_DFF = D * 4
    NX_DFF = int(D * 4 * 2 / 3)

    n_bins = 16  # 将序列分成 16 个区间
    bin_size = SEQ // n_bins
    n_trials = 10

    results = {}
    for name, model_cls, dff in [
        ("Baseline", BaselineGPT, BL_DFF),
        ("NEXUS (全组件)", NexusGPT, NX_DFF),
    ]:
        torch.manual_seed(42)
        np.random.seed(42)
        model = model_cls(V, D, L, H, dff, SEQ).to(DEVICE).eval()
        bin_losses = np.zeros(n_bins)

        with torch.no_grad():
            for trial in range(n_trials):
                # 从验证集采样
                starts = np.random.randint(0, len(val_data) - SEQ - 1, size=BATCH)
                x = torch.stack([
                    torch.from_numpy(val_data[s:s + SEQ].astype(np.int64))
                    for s in starts
                ]).to(DEVICE)
                y = torch.stack([
                    torch.from_numpy(val_data[s + 1:s + SEQ + 1].astype(np.int64))
                    for s in starts
                ]).to(DEVICE)

                logits, _ = model(x)

                for b in range(n_bins):
                    start = b * bin_size
                    end = (b + 1) * bin_size
                    bin_logits = logits[:, start:end, :].reshape(-1, V)
                    bin_targets = y[:, start:end].reshape(-1)
                    loss = F.cross_entropy(bin_logits, bin_targets).item()
                    bin_losses[b] += loss / n_trials

        # 线性回归计算斜率
        positions = np.arange(n_bins)
        slope, intercept = np.polyfit(positions, bin_losses, 1)

        results[name] = {
            "bin_losses": bin_losses.tolist(),
            "slope": slope,
            "intercept": intercept,
        }

        print(f"\n  [{name}]")
        print(f"  Position:  ", end="")
        for b in range(0, n_bins, 2):
            print(f" {b * bin_size:>4d}-{(b + 1) * bin_size:>4d}", end="")
        print()
        print(f"  Loss:      ", end="")
        for b in range(0, n_bins, 2):
            print(f"  {bin_losses[b]:>8.4f}", end="")
        print()
        print(f"  线性斜率: {slope:+.6f}/bin ({'下降 ✅' if slope < -0.001 else '上升 ⚠️' if slope > 0.001 else '稳定'})")

        del model
        torch.cuda.empty_cache()

    # 对比斜率
    bl_slope = results["Baseline"]["slope"]
    nx_slope = results["NEXUS (全组件)"]["slope"]
    print(f"\n  --- 在线学习斜率对比（负值 = 越到后面越好）---")
    print(f"  Baseline 斜率: {bl_slope:+.6f}")
    print(f"  NEXUS    斜率: {nx_slope:+.6f}")

    if nx_slope < bl_slope:
        print(f"  ✅ NEXUS 的 loss 下降趋势更明显（斜率更负）")
        print(f"     TTT 在 forward 中的在线适应正在生效！")
    else:
        print(f"  ⚠️ Baseline 的 loss 下降趋势反而更好")

    return results


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("  TTT 在线适应能力专项测试")
    print(f"  Device: {DEVICE}")
    print(f"  Model: d=512, L=4, seq=1024 (全组件: DiffAttn+MLA+TTT+SwiGLU)")
    print("  注意: 随机权重即可测试在线学习能力")
    print("=" * 70)

    t0 = time.time()

    # TEST 1: Per-Position Loss 衰减
    r1 = test_per_position_loss()

    # TEST 2: Induction Pattern
    r2 = test_induction_pattern()

    # TEST 3: 真实数据上的 Loss 斜率
    r3 = test_loss_slope()

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  全部完成！总耗时: {total:.0f}s ({total / 60:.1f} min)")
    print(f"{'=' * 70}")

    # 汇总
    print(f"\n  === TTT 在线适应能力评估汇总 ===")
    print(f"  测试的是 forward pass 中的即时学习，不是训练优化。")
    print(f"  TTT 的价值在于：模型无需重新训练，在推理时就能适应新模式。")
    print(f"  如果以上测试显示 TTT 有在线学习信号，")
    print(f"  那么在更大模型 + 更长序列（4096+）上，这个优势会被放大。")


if __name__ == "__main__":
    main()
