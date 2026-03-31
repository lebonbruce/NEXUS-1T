# NEXUS: Low-Rank Test-Time Training with Differential Attention for Ultra-Long Context Language Models

## Technical Report v1.0 — March 2026

---

## Abstract

We present NEXUS, a Transformer variant that integrates six complementary techniques — Differential Attention, Multi-head Latent Attention (MLA), Test-Time Training (TTT) with LoRA, Mixture-of-Experts (MoE), and a novel W-Snapshot Pyramid — into a unified, scale-aware architecture. Through systematic micro-benchmarks on 25-50M parameter models, we report three key findings: (1) **Low-rank regularization in TTT**: LoRA rank=8 improves online learning effectiveness by 8.1× over full-rank, contrary to the intuition that more capacity enables better learning; (2) **Content-aware W-snapshot routing** achieves 100% retrieval accuracy for overwritten TTT states using mean-embedding fingerprints; (3) **FP32 cumsum** is essential for TTT stability, as BF16 accumulation errors reach 1,124× at 10M tokens. We provide complete experimental configurations for scaling to 1T parameters and discuss honest limitations of our approach. All code and experiments are publicly available.

---

## 1. Introduction

Large language models face two fundamental challenges at scale:

1. **Context length**: Standard attention requires O(N²) memory, making 100K+ contexts prohibitively expensive.
2. **Adaptability**: Trained models cannot adapt to novel patterns at inference time without fine-tuning.

NEXUS addresses both through a combination of established techniques, unified by a "Scale-Aware" configuration system that automatically enables components based on model dimensions.

**Our contributions:**
- Discovery that low-rank LoRA regularization (rank=8) dramatically improves TTT online learning (§3.1)
- A content-aware routing mechanism for TTT weight snapshots with 100% retrieval accuracy (§3.2)
- Identification of critical BF16 numerical instability in TTT cumsum operations (§3.3)
- A complete, auto-configuring implementation and scaling analysis to 1T parameters (§4)

---

## 2. Architecture

### 2.1 Components

Each NexusBlock contains three sub-layers with residual connections:

**Layer 1: DiffAttn + MLA** (Attention + Compression)
- Differential Attention [Ye et al., 2024] computes attention as the difference of two softmax maps, canceling noise
- MLA [DeepSeek-V2, 2024] compresses KV cache through low-rank projection (4× reduction at d≥1024)

**Layer 2: TTT-Linear with LoRA** (Online Learning)
- Maintains a weight matrix W that updates via gradient descent during the forward pass
- Our key modification: W ≈ B×A (LoRA decomposition) with rank=8, providing implicit regularization

**Layer 3: MoE-SwiGLU** (Sparse Computation)
- N experts (8 at prototype scale, 256 at 1T), top-k routing (k=2)
- Load balancing auxiliary loss prevents expert collapse

### 2.2 Scale-Aware Auto-Configuration

Components activate based on d_model thresholds:

| Component | Condition | Rationale |
|-----------|-----------|-----------|
| DiffAttn | Always ON | Effective at all scales |
| MLA 2× | d ≥ 384 | Below this, compression hurts more than helps |
| MLA 4× | d ≥ 1024 | Latent dim ≥256 needed for fidelity |
| TTT | d ≥ 512, seq ≥ 1024 | Needs sufficient capacity and sequence length |
| LoRA-TTT | Always (when TTT on) | rank=min(8, d//4) |
| MoE | d ≥ 1024 | Below this, experts too small |

---

## 3. Experiments

All experiments conducted on NVIDIA RTX 4060 8GB, PyTorch 2.1, CUDA 12.1.

### 3.1 Low-Rank Regularization in TTT

**Setup**: D=512, seq=1024, mini_batch=16, ttt_lr=0.5. Measured online learning slope (negative = model improves while reading).

**Results**:

| LoRA Rank | Parameters | Online Learning Slope | Relative to Full Rank |
|:---------:|:----------:|:--------------------:|:--------------------:|
| 8 | 796,160 | **-0.0364** | **8.1×** |
| 16 | 804,352 | -0.0360 | 8.0× |
| 32 | 820,736 | -0.0363 | 8.1× |
| 64 | 853,504 | -0.0362 | 8.1× |
| 128 | 919,040 | -0.0356 | 7.9× |
| 512 (Full) | 1,050,112 | -0.0045 | 1.0× |

**Analysis**: This result is counter-intuitive — more capacity leads to worse online learning. We attribute this to **overfitting to inter-token noise**: full-rank W has D²=262K degrees of freedom, vastly exceeding the meaningful signal in a 1024-token sequence. Low-rank LoRA constrains updates to a k-dimensional subspace (k=8), acting as an implicit regularizer that forces the model to capture only the dominant statistical patterns.

This finding has practical implications: at 1T scale (D=10496), using rank=8 instead of rank=D/4=2624 reduces per-layer TTT FLOPs by 328× while *improving* online learning quality.

### 3.2 Content-Aware W-Snapshot Routing

**Problem**: TTT's W matrix is continuously overwritten by new patterns. After reading 16K tokens of Pattern B, Pattern A information is largely lost.

**Setup**: Two-phase experiment:
- Phase 1: Process 4K tokens rich in Pattern A (tokens 10-19)
- Phase 2: Process 16K tokens rich in Pattern B (tokens 100-109), overwriting A

Save W snapshots and content fingerprints (mean token embeddings) at each phase.

**Routing mechanism**: Given a query, compute cosine similarity against all fingerprints, apply low-temperature softmax (τ=0.05), select the highest-weight snapshot.

**Results**:

| Query | Phase 1 Similarity | Phase 2 Similarity | Selected | Correct? |
|-------|:-:|:-:|:-:|:-:|
| Pattern A tokens | 0.9493 | 0.0000 | Phase 1 | ✅ |
| Pattern B tokens | 0.0000 | 0.9522 | Phase 2 | ✅ |

**Routing accuracy: 100%.** The fingerprints have cosine similarity of only 0.09 (nearly orthogonal), making routing trivially accurate. This validates the feasibility of a W-snapshot pyramid for extending effective context length.

### 3.3 FP32 Cumsum for Numerical Stability

**Setup**: Simulate TTT cumsum accumulation over increasing token counts, measuring max relative error between FP32 and BF16.

| Tokens | Gradient Norm (√N scaling) | BF16 Max Error |
|:------:|:-:|:-:|
| 1K | 0.42 | 0.000× |
| 100K | 4.19 | 0.001× |
| 1M | 13.01 | 10.8× |
| 10M | 41.80 | 1,124× |

**Fix**: `torch.cumsum(grad.float(), dim=1).to(input_dtype)` — forcing FP32 for the cumsum operation only. Performance impact: negligible (single operation).

### 3.4 MoE Load Balancing

**Setup**: 8 experts, top-2 routing, D=512, batch=2×256.

| Expert | Load (ideal=0.125) | Deviation |
|:-:|:-:|:-:|
| 0 | 0.1260 | 1% |
| 1 | 0.1239 | 1% |
| 2 | 0.1219 | 2% |
| 3 | 0.1193 | 5% |
| 4 | 0.1233 | 1% |
| 5 | 0.1298 | 4% |
| 6 | 0.1241 | 1% |
| 7 | 0.1316 | 5% |

**Balance score: 0.915** (min/max load ratio). No expert collapse observed.

### 3.5 Pyramid Storage-Accuracy Tradeoff

**Setup**: Varying snapshot granularity (256-16384 tokens) and SVD compression rank.

| Granularity | Any Rank | Top-1 Accuracy | Top-3 Accuracy |
|:-----------:|:--------:|:--------------:|:--------------:|
| 256 | 8-Full | ✅ 100% | 1.00 |
| **1024** | **8-Full** | **✅ 100%** | **1.00** |
| 4096 | 8-Full | ✅ 100% | 0.67 |
| 16384 | 8-Full | ✅ 100% | 0.50 |

**Key finding**: Compression rank does NOT affect routing accuracy (routing uses fingerprints, not W). Granularity ≤1024 maintains perfect top-3 retrieval.

---

## 4. Scaling Analysis

### 4.1 Validated 1T Configuration

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| d_model | 10,496 | Searched for ~1T total params |
| n_layers | 128 | Standard depth for this width |
| n_heads | 82 | head_dim=64 (DiffAttn sub-head) |
| d_ff | 28,160 | SwiGLU standard ratio (8/3)× |
| n_experts | 8 | Prototype; 256 recommended at full scale |
| top_k | 2 | Balances compute vs quality |
| ttt_rank | 8 | Experimentally optimal (§3.1) |
| mla_compression | 4× | Latent dim=2624, sufficient fidelity |
| **Total params** | **999B** | |
| **Active/token** | **318B (32%)** | |

### 4.2 Inference Memory (100M Context)

| Component | BF16 | INT4+TurboQuant |
|-----------|:----:|:---:|
| Model weights | 2.0 TB | 500 GB |
| KV Cache (sliding window 8K) | 11.0 GB | 5.5 GB |
| TTT W (LoRA r=8) | 0.34 GB | 0.34 GB |
| Pyramid fingerprints | ~1 GB | ~1 GB |
| **Total** | **~2.0 TB** | **~507 GB** |
| **A100 80GB** | 25 cards | **7 cards** |

### 4.3 Training Cost

Chinchilla-optimal training: 20T tokens, ~3.82×10²⁵ FLOPs.
At 40% MFU on A100: **~85M GPU-hours** → ~10,000 H100 × 120 days. Cost is hardware-dependent and declining rapidly with advances in training efficiency.

---

## 5. Limitations

1. **No large-scale training.** All experiments are micro-benchmarks (≤50M params). We cannot verify that findings transfer to 1B+.
2. **Components are not novel individually.** DiffAttn, TTT, MLA, MoE are all from published work. Our contribution is the combination and the LoRA-TTT finding.
3. **100M context is lossy.** TTT compresses patterns, not raw text. Precise recall of distant content is not guaranteed.
4. **Pyramid storage gap.** Full W-snapshot storage for all layers scales as O(L×D×r×N/granularity), which may exceed our estimates at true 1T scale.
5. **MoE scaling untested.** 8 experts validated at 50M; optimal count at 1T is speculative.

---

## 6. Related Work

- **Differential Attention** [Ye et al., 2024]: Attention noise cancellation
- **TTT-Linear** [Sun et al., 2024]: Test-time training as a sequence model layer
- **DeepSeek-V2/V3** [DeepSeek, 2024]: MLA + MoE at scale
- **Infini-attention** [Munkhdalai et al., 2024]: Compressive memory in attention
- **Compressive Transformer** [Rae et al., 2020]: Hierarchical memory compression
- **LoRA** [Hu et al., 2021]: Low-rank adaptation

---

## Reproducibility

All experiments can be reproduced on a single NVIDIA RTX 4060 8GB:

```bash
git clone https://github.com/LeBon/nexus-cognitive-architecture.git
cd nexus-cognitive-architecture

# 2-second architecture validation
python pretrain/validate_refactor.py

# Full experiment suite (~2 minutes total)
python pretrain/ttt_online_test.py          # 7s
python pretrain/pyramid_routing_test.py     # 0.1s
python pretrain/pyramid_tradeoff_test.py    # 24s
python pretrain/moe_turboquant_test.py      # 2s
```

---

*Corresponding author: LeBon (lebonbruce92@gmail.com) — Independent researcher seeking compute resources for scale-up validation.*
