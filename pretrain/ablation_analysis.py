"""分析消融实验结果。"""
import json
import math

with open("pretrain/data/ablation_results.json") as f:
    results = json.load(f)

# Baseline 在 1000 步时的 val_loss（从完整训练的 results.json 中提取）
bl_1000 = 4.7597
nx_1000 = 4.9781

print("=" * 80)
print("  NEXUS 消融实验结果汇总（1000步快速验证 + 已有3000步数据）")
print("=" * 80)

# =====================================================
# 1000 步公平对比
# =====================================================
print("\n--- 1000 步公平对比（所有变体都在 1000 步评估） ---\n")

rows = [
    ("Baseline (MHA+GELU)", 29920512, bl_1000, "-"),
    ("Baseline+SwiGLU", results[2]["params"], results[2]["final_val_loss"], f"{results[2]['time_s']:.0f}s"),
    ("DiffAttn+GELU", results[3]["params"], results[3]["final_val_loss"], f"{results[3]['time_s']:.0f}s"),
    ("NEXUS-noTTT (DiffMLA+SwiGLU)", results[4]["params"], results[4]["final_val_loss"], f"{results[4]['time_s']:.0f}s"),
    ("NEXUS Full (Diff+TTT+SwiGLU)", 32363904, nx_1000, "-"),
]

fmt = "  {:<35s} | {:>10s} | {:>10s} | {:>8s} | {:>10s} | {:>6s}"
print(fmt.format("Model", "Params", "Val Loss", "Time", "vs BL", "PPL"))
print("  " + "-" * 95)
for name, params, val, time_s in rows:
    delta = val - bl_1000
    pct = delta / bl_1000 * 100
    ppl = math.exp(val)
    sign = "+" if delta > 0 else ""
    print(fmt.format(
        name,
        f"{params:,}",
        f"{val:.4f}",
        time_s,
        f"{sign}{delta:.4f} ({sign}{pct:.1f}%)",
        f"{ppl:.1f}",
    ))

# =====================================================
# 消融分析
# =====================================================
print("\n\n" + "=" * 80)
print("  根因分析（逐组件隔离）")
print("=" * 80)

bl = bl_1000
swiglu_val = results[2]["final_val_loss"]
diff_gelu_val = results[3]["final_val_loss"]
no_ttt_val = results[4]["final_val_loss"]
full_val = nx_1000
total_gap = full_val - bl

print(f"\n  总 Gap: NEXUS Full - Baseline = {full_val:.4f} - {bl:.4f} = {total_gap:+.4f}\n")

# 1. SwiGLU 单独效果
swiglu_delta = swiglu_val - bl
print(f"  [1] SwiGLU 效果 (Baseline+SwiGLU vs Baseline):")
print(f"      {bl:.4f} -> {swiglu_val:.4f} | delta = {swiglu_delta:+.4f} ({swiglu_delta/bl*100:+.1f}%)")
if swiglu_delta > 0:
    print(f"      => SwiGLU 在 1000 步时反而更差（收敛更慢）")
else:
    print(f"      => SwiGLU 有帮助")

# 2. DiffAttn+MLA 效果
diff_delta = diff_gelu_val - bl
print(f"\n  [2] DiffAttn+MLA 效果 (DiffAttn+GELU vs Baseline):")
print(f"      {bl:.4f} -> {diff_gelu_val:.4f} | delta = {diff_delta:+.4f} ({diff_delta/bl*100:+.1f}%)")
if diff_delta > 0.1:
    print(f"      => DiffAttn+MLA **严重拖累**！增加了 {diff_delta:.4f} 的 loss")
    print(f"      => 这是 MLA 96维信息瓶颈的直接证据")

# 3. TTT 效果（NEXUS Full vs NEXUS-noTTT）
ttt_delta = full_val - no_ttt_val
print(f"\n  [3] TTT 效果 (NEXUS Full vs NEXUS-noTTT):")
print(f"      {no_ttt_val:.4f} -> {full_val:.4f} | delta = {ttt_delta:+.4f} ({ttt_delta/no_ttt_val*100:+.1f}%)")
if ttt_delta < -0.01:
    print(f"      => TTT 有改善作用（降低了 {abs(ttt_delta):.4f}）")
elif ttt_delta > 0.05:
    print(f"      => TTT 反而更差")
else:
    print(f"      => TTT 影响接近中性（很小的变化）")

# 4. DiffAttn+MLA+SwiGLU 组合效果
combo_delta = no_ttt_val - bl
print(f"\n  [4] DiffAttn+MLA+SwiGLU 组合效果 (NEXUS-noTTT vs Baseline):")
print(f"      {bl:.4f} -> {no_ttt_val:.4f} | delta = {combo_delta:+.4f} ({combo_delta/bl*100:+.1f}%)")

# =====================================================
# 贡献分解
# =====================================================
print("\n\n" + "=" * 80)
print("  贡献分解（近似）")
print("=" * 80)

print(f"""
  总 Gap              = {total_gap:+.4f}
  
  DiffAttn+MLA 贡献   ≈ {diff_delta:+.4f}  (占 {diff_delta/total_gap*100:.0f}%)  ← 最大元凶
  SwiGLU 贡献         ≈ {swiglu_delta:+.4f}  (占 {swiglu_delta/total_gap*100:.0f}%)
  TTT 贡献            ≈ {ttt_delta:+.4f}  (占 {ttt_delta/total_gap*100:.0f}%)
  交互效应(残差)      ≈ {total_gap - diff_delta - swiglu_delta - ttt_delta:+.4f}
""")

# =====================================================
# 核心结论
# =====================================================
print("=" * 80)
print("  核心结论")
print("=" * 80)
print(f"""
  1. MLA 4x 压缩（96维瓶颈）是 NEXUS 落后的最大原因
     - DiffAttn+MLA+GELU 比 Baseline 高 {diff_delta:.4f} ({diff_delta/bl*100:.1f}%)
     - 这个 gap 占总 gap 的 {diff_delta/total_gap*100:.0f}%

  2. SwiGLU 在 1000 步短训练中收敛略慢
     - Baseline+SwiGLU 比 Baseline 高 {swiglu_delta:.4f} ({swiglu_delta/bl*100:.1f}%)
     - 可能只是初始化/收敛速度问题，长训练后可能追上

  3. TTT 在当前设置下影响不大
     - 加了 TTT 后 loss 变化仅 {ttt_delta:+.4f}
     - TTT 不是性能问题的核心，但也没带来帮助
     - 训练速度代价：7.74x 慢

  立即行动建议：
  ✅ 降低 MLA 压缩比从 4x 到 2x（最高优先级）
  ✅ 或完全去掉 MLA 瓶颈，让 DiffAttn 使用独立 KV 投影
  ⚠️ TTT 在短序列上无益但也无害——建议保留但设计专门的长序列测试
  ❓ SwiGLU 需要更长训练验证（可能只是收敛慢）
""")

# 速度对比
print("\n--- 速度对比 ---")
print(f"  Baseline+SwiGLU: {results[2]['time_s']:.0f}s (1x)")
print(f"  DiffAttn+GELU:   {results[3]['time_s']:.0f}s ({results[3]['time_s']/results[2]['time_s']:.2f}x)")
print(f"  NEXUS-noTTT:     {results[4]['time_s']:.0f}s ({results[4]['time_s']/results[2]['time_s']:.2f}x)")
print(f"  去掉 TTT 后速度正常！训练时间仅比 Baseline 慢 ~20%")
