"""深度分析 3000 步预训练对比结果 + 参数量精确拆解。"""
import json
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 参数量精确拆解
def count_component_params():
    """精确计算每个组件的参数量。"""
    D = 384  # d_model
    H = 6   # n_heads
    V = 50257  # vocab_size
    L = 6   # n_layers
    
    print("=" * 70)
    print("  参数量精确拆解")
    print("=" * 70)
    
    # === Baseline ===
    tok_emb = V * D  # 19,298,688 (weight tied with head)
    
    # CausalSelfAttention per layer
    qkv_proj = D * 3 * D  # 442,368
    out_proj = D * D       # 147,456
    # RoPE: 0 (纯计算)
    attn_total = qkv_proj + out_proj  # 589,824
    
    # GELU MLP per layer
    d_ff_bl = 1536
    gelu_fc1 = D * d_ff_bl  # 589,824
    gelu_fc2 = d_ff_bl * D  # 589,824
    gelu_total = gelu_fc1 + gelu_fc2  # 1,179,648
    
    # RMSNorm per layer (ln1 + ln2)
    rms_per_layer = 2 * D  # 768
    
    # Final RMSNorm
    rms_final = D  # 384
    
    bl_per_layer = attn_total + gelu_total + rms_per_layer
    bl_total = tok_emb + L * bl_per_layer + rms_final
    # head is weight-tied, not counted separately
    
    print(f"\n  --- Baseline GPT ---")
    print(f"  Token Embedding:   {tok_emb:>12,} (weight tied)")
    print(f"  Per Layer:")
    print(f"    CausalSelfAttn:  {attn_total:>12,} (qkv={qkv_proj:,} + out={out_proj:,})")
    print(f"    GELU MLP:        {gelu_total:>12,} (fc1={gelu_fc1:,} + fc2={gelu_fc2:,})")
    print(f"    RMSNorm x2:      {rms_per_layer:>12,}")
    print(f"    Layer total:     {bl_per_layer:>12,}")
    print(f"  x {L} layers:       {L*bl_per_layer:>12,}")
    print(f"  Final RMSNorm:     {rms_final:>12,}")
    print(f"  TOTAL:             {bl_total:>12,}")
    
    # === NEXUS ===
    d_ff_nx = 1024
    hd = D // H // 2  # 32 (DiffAttn sub-head dim)
    d_kv_latent = D // 4  # 96
    
    # DiffAttnMLA per layer
    q_proj = D * D  # 147,456
    kv_down = D * d_kv_latent  # 36,864
    k_up = d_kv_latent * D  # 36,864
    v_up = d_kv_latent * D  # 36,864
    diff_out = D * D  # 147,456
    lambda_params = 4 * hd  # 128
    subln = 2 * hd  # 64
    # RoPE: 0
    diff_attn_total = q_proj + kv_down + k_up + v_up + diff_out + lambda_params + subln
    
    # TTT-Linear per layer
    theta_proj = D * D  # 147,456
    ttt_W = D * D  # 147,456
    lr_gate = D * D + D  # 147,840 (Linear + bias)
    output_gate = D * D + D  # 147,840
    ttt_norm = D  # 384
    ttt_total = theta_proj + ttt_W + lr_gate + output_gate + ttt_norm
    
    # SwiGLU FFN per layer
    swiglu_w1 = D * d_ff_nx  # 393,216
    swiglu_w2 = d_ff_nx * D  # 393,216
    swiglu_w3 = D * d_ff_nx  # 393,216
    swiglu_total = swiglu_w1 + swiglu_w2 + swiglu_w3
    
    # RMSNorm per layer (ln1 + ln2 + ln3)
    rms_nx_per_layer = 3 * D  # 1152
    
    nx_per_layer = diff_attn_total + ttt_total + swiglu_total + rms_nx_per_layer
    nx_total = tok_emb + L * nx_per_layer + rms_final
    
    print(f"\n  --- NEXUS GPT ---")
    print(f"  Token Embedding:   {tok_emb:>12,} (weight tied)")
    print(f"  Per Layer:")
    print(f"    DiffAttn+MLA:    {diff_attn_total:>12,}")
    print(f"      q_proj:        {q_proj:>12,}")
    print(f"      kv_down:       {kv_down:>12,}")
    print(f"      k_up:          {k_up:>12,}")
    print(f"      v_up:          {v_up:>12,}")
    print(f"      out_proj:      {diff_out:>12,}")
    print(f"      lambda+subln:  {lambda_params+subln:>12,}")
    print(f"    TTT-Linear:      {ttt_total:>12,}")
    print(f"      theta_proj:    {theta_proj:>12,}")
    print(f"      W:             {ttt_W:>12,}")
    print(f"      lr_gate:       {lr_gate:>12,}")
    print(f"      output_gate:   {output_gate:>12,}")
    print(f"      norm:          {ttt_norm:>12,}")
    print(f"    SwiGLU FFN:      {swiglu_total:>12,}")
    print(f"      w1+w2+w3:      {swiglu_w1:,}+{swiglu_w2:,}+{swiglu_w3:,}")
    print(f"    RMSNorm x3:      {rms_nx_per_layer:>12,}")
    print(f"    Layer total:     {nx_per_layer:>12,}")
    print(f"  x {L} layers:       {L*nx_per_layer:>12,}")
    print(f"  Final RMSNorm:     {rms_final:>12,}")
    print(f"  TOTAL:             {nx_total:>12,}")
    
    # === 对比 ===
    print(f"\n  --- 参数量对比 ---")
    print(f"  Baseline Attn:   {L*attn_total:>12,}")
    print(f"  NEXUS DiffAttn:  {L*diff_attn_total:>12,} ({(L*diff_attn_total - L*attn_total):+,})")
    print(f"  Baseline GELU:   {L*gelu_total:>12,}")
    print(f"  NEXUS SwiGLU:    {L*swiglu_total:>12,} ({(L*swiglu_total - L*gelu_total):+,})")
    print(f"  NEXUS TTT (NEW): {L*ttt_total:>12,} (纯新增)")
    print(f"  Total gap:       {nx_total-bl_total:>12,} ({(nx_total-bl_total)/bl_total*100:.1f}%)")
    
    # 每个组件占 NEXUS 的比例
    print(f"\n  --- NEXUS 参数分布 ---")
    non_emb = nx_total - tok_emb - rms_final
    print(f"  DiffAttn+MLA:  {L*diff_attn_total/non_emb*100:5.1f}%  ({L*diff_attn_total:,})")
    print(f"  TTT-Linear:    {L*ttt_total/non_emb*100:5.1f}%  ({L*ttt_total:,})")
    print(f"  SwiGLU FFN:    {L*swiglu_total/non_emb*100:5.1f}%  ({L*swiglu_total:,})")
    print(f"  RMSNorm:       {L*rms_nx_per_layer/non_emb*100:5.1f}%  ({L*rms_nx_per_layer:,})")


def analyze_convergence():
    """分析收敛特性。"""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.json")) as f:
        r = json.load(f)
    
    b = r['baseline']
    n = r['nexus']
    
    print("\n\n" + "=" * 70)
    print("  收敛特性深度分析")
    print("=" * 70)
    
    # Gap 趋势分析
    print("\n  --- Gap 趋势 ---")
    gaps = []
    for bv, nv in zip(b['val_losses'], n['val_losses']):
        gap = nv[1] - bv[1]
        gaps.append((bv[0], gap, gap/bv[1]*100))
    
    # 前半 vs 后半的 gap
    first_half = [g[2] for g in gaps[:6]]
    second_half = [g[2] for g in gaps[6:]]
    print(f"  前 1500 步平均 gap: {sum(first_half)/len(first_half):.1f}%")
    print(f"  后 1500 步平均 gap: {sum(second_half)/len(second_half):.1f}%")
    
    if sum(second_half)/len(second_half) > sum(first_half)/len(first_half):
        print("  ⚠️ Gap 在扩大！NEXUS 收敛不如 Baseline")
    else:
        print("  ✅ Gap 在缩小，NEXUS 有追赶趋势")
    
    # 最后 1000 步的 loss 下降速度
    print("\n  --- 最后 1000 步下降速度 ---")
    b_vals = dict(b['val_losses'])
    n_vals = dict(n['val_losses'])
    
    b_2000 = b_vals[2000]
    b_3000 = b_vals[3000]
    n_2000 = n_vals[2000]
    n_3000 = n_vals[3000]
    
    b_drop = b_2000 - b_3000
    n_drop = n_2000 - n_3000
    
    print(f"  BL: {b_2000:.4f} -> {b_3000:.4f} (降 {b_drop:.4f})")
    print(f"  NX: {n_2000:.4f} -> {n_3000:.4f} (降 {n_drop:.4f})")
    
    if n_drop > b_drop:
        print(f"  ✅ NEXUS 后期下降更快 ({n_drop:.4f} vs {b_drop:.4f})")
        # 外推：如果保持这个下降率，多少步后追平？
        gap_at_3000 = n_3000 - b_3000
        rate_diff = n_drop - b_drop  # NEXUS 每 1000 步多降多少
        if rate_diff > 0:
            steps_to_catch = gap_at_3000 / rate_diff * 1000
            print(f"  📊 以此下降率外推，约 {steps_to_catch:.0f} 步后追平（需验证）")
    else:
        print(f"  ❌ Baseline 后期仍下降更快")
    
    # Train vs Val gap（过拟合检测）
    print("\n  --- 过拟合检测 ---")
    b_train_last = dict(b['train_losses'])[2950]
    n_train_last = dict(n['train_losses'])[2950]
    
    b_overfit = b_3000 - b_train_last
    n_overfit = n_3000 - n_train_last
    
    print(f"  BL: train={b_train_last:.4f} val={b_3000:.4f} gap={b_overfit:+.4f}")
    print(f"  NX: train={n_train_last:.4f} val={n_3000:.4f} gap={n_overfit:+.4f}")
    
    if abs(n_overfit) < abs(b_overfit):
        print(f"  ✅ NEXUS 泛化更好（train-val gap 更小）")
    else:
        print(f"  ⚠️ NEXUS 泛化不如 Baseline")
    
    # 关键发现
    print("\n\n" + "=" * 70)
    print("  关键发现总结")
    print("=" * 70)
    
    print(f"""
  1. NEXUS 在 3000 步 val_loss 落后 {n_3000-b_3000:.4f} ({(n_3000-b_3000)/b_3000*100:.1f}%)
  2. NEXUS 多 {n['params']-b['params']:,} 参数 (+{(n['params']-b['params'])/b['params']*100:.1f}%)，
     但 loss 更高 → 参数效率更差
  3. NEXUS 训练慢 {n['time']/b['time']:.2f}x，主要由 TTT 的 cumsum + matmul 导致
  4. Gap 趋势：{'扩大' if sum(second_half)/len(second_half) > sum(first_half)/len(first_half) else '缩小'}
  5. 后期下降率：NEXUS {'更快' if n_drop > b_drop else '更慢'}
  6. PPL: BL={math.exp(b_3000):.1f} vs NX={math.exp(n_3000):.1f}
  
  核心问题：
    - TTT 层增加了 ~3.5M 参数 + 7x 训练时间
    - 但没有带来 loss 改善
    - DiffAttn+MLA 参数更少（kv 低秩压缩），可能表达力不足
    - 需要消融实验来分离 TTT 和 DiffAttn+MLA 的各自贡献
""")


if __name__ == "__main__":
    count_component_params()
    analyze_convergence()
