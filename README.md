# NEXUS: Structure for Compute, Memory for Intelligence

**English** | [中文](README_CN.md)

**A cognitive architecture that trades structure for compute and memory for intelligence.**

> 个人开发者项目。没有 $5B 去训练 1T 模型，但用科学方法验证了架构在 scale-up 时的鲁棒性。
> 所有实验数据公开，所有代码可复现。

---

## What is NEXUS?

NEXUS 是一个融合了 6 项前沿技术的 Transformer 变体架构，目标是实现**超长上下文**（理论 100M tokens）和**推理时在线学习**：

| 组件 | 来源 | 作用 | 论文 |
|------|------|------|------|
| **Differential Attention** | Microsoft Research, 2024 | 注意力降噪，消除无关 token 干扰 | [arXiv:2410.05258](https://arxiv.org/abs/2410.05258) |
| **Multi-head Latent Attention (MLA)** | DeepSeek, 2024 | KV Cache 压缩 4x | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| **Test-Time Training (TTT)** | Stanford, 2024 | 推理时在线学习，无需重训练 | [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) |
| **LoRA-TTT** (本项目发现) | — | 低秩正则化让在线学习效果提升 8x | 见下方实验 |
| **MoE-SwiGLU** | Google/Meta | 稀疏激活，1T 参数只用 32% 计算 | [Switch Transformer](https://arxiv.org/abs/2101.03961) |
| **W-Snapshot Pyramid** (本项目提出) | — | 分层压缩记忆实现超长上下文 | 见下方实验 |

### Architecture Diagram

```
Input → Embedding → [NexusBlock × N] → RMSNorm → LM Head → Output

NexusBlock:
  ┌─ RMSNorm → DiffAttn + MLA → Residual ─┐
  │                                         │
  ├─ RMSNorm → TTT-Linear (LoRA) → Residual ─┤  (d≥512, seq≥1024)
  │                                         │
  └─ RMSNorm → SwiGLU / MoE-SwiGLU → Residual ─┘
```

---

## Key Findings (All Experimentally Verified)

### 1. 🏆 Low-Rank Regularization in TTT (Most Significant)

**Discovery**: LoRA rank=8 outperforms full-rank TTT by **8x** in online learning.

```
Rank |  TTT Params | Online Learning Slope | vs Full Rank
-----|-------------|----------------------|-------------
   8 |     796,160 |            -0.036392 |    8.1x ⭐
  16 |     804,352 |            -0.036026 |    8.0x
  32 |     820,736 |            -0.036272 |    8.1x
  64 |     853,504 |            -0.036220 |    8.1x
 128 |     919,040 |            -0.035597 |    7.9x
Full |   1,050,112 |            -0.004475 |    1.0x
```

**Why**: Full-rank W has D²=262K degrees of freedom — it overfits to random noise between tokens. Low-rank forces the model to only update along the most important semantic directions. **TTT is not a memory store — it's a pattern filter.**

### 2. Content-Aware W-Snapshot Routing (100% Accuracy)

**Problem**: TTT's W matrix overwrites old patterns when learning new ones.

**Solution**: Save W snapshots + content fingerprints at regular intervals. Route queries to the correct snapshot using cosine similarity.

```
Query Pattern A:
  → Phase1 W (A still intact): weight=1.0000 ⭐
  → Phase2 W (A overwritten):  weight=0.0000

Query Pattern B:
  → Phase1 W: weight=0.0000
  → Phase2 W (B just learned): weight=1.0000 ⭐

Routing accuracy: 100%
```

**Key insight**: Content fingerprints (mean embeddings) have cos=0.09 — nearly orthogonal. This makes routing trivially accurate.

### 3. FP32 Cumsum Eliminates BF16 Numerical Drift

TTT uses cumulative sums that accumulate errors in BF16:

```
Token Count |    BF16 Error | Action
------------|---------------|-------
      1,000 |        0.001x | Safe
    100,000 |        1.05x  | Warning
  1,000,000 |       10.8x   | Dangerous
 10,000,000 |     1,124x    | ❌ Training collapse

Fix: torch.cumsum(grad.float(), dim=1).to(input_dtype)
Cost: ~0% (only cumsum op in FP32)
```

### 4. MoE Expert Load Balancing

8 experts with top-2 routing achieve 0.915 balance score (1.0 = perfect):

```
Expert 0: 0.1260 (deviation 1%)
Expert 1: 0.1239 (deviation 1%)
Expert 2: 0.1219 (deviation 2%)
Expert 3: 0.1193 (deviation 5%)
...
Balance: 0.915 ✅
```

### 5. Scale-Aware Auto-Configuration

All components automatically enable/disable based on model dimensions:

```python
# models.py — automatic decisions
MLA:  OFF (d<384) → 2x (384-1023) → 4x (d≥1024)
TTT:  OFF (d<512 or seq<1024) → LoRA rank=min(8, d//4)
MoE:  OFF (d<1024) → 8 experts (d≥1024) → 256 experts (1T)
```

---

## 1T Model Configuration (Theoretical)

Based on our micro-benchmark validated parameters:

```
╔═══════════════════════════════════╗
║  d_model     =  10,496           ║
║  n_layers    =     128           ║
║  n_heads     =      82           ║
║  d_ff        =  28,160           ║
║  n_experts   =       8 (top-2)   ║
║  ttt_rank    =       8 (LoRA)    ║
║  Total Params = ~1T              ║
║  Active/token = ~318B (32%)      ║
╚═══════════════════════════════════╝

Inference (100M context, INT4):
  Model:     500 GB
  KV Cache:  5.5 GB (sliding window 8K + MLA 4x)  
  TTT W:     0.34 GB
  Pyramid:   263 GB (fingerprints only)
  Total:     ~769 GB → 10× A100 80GB

Training: ~10,000× H100, 120 days, ~$5B
```

> **Honest disclaimer**: 100M context is a "lossy semantic compression window." It preserves logical structure but loses verbatim details. Think of it as reading 100 books and remembering the key arguments, not the exact sentences.

---

## Limitations (Honest Assessment)

1. **No large-scale training results.** All findings are from micro-benchmarks (25-50M parameters, <1024 tokens). We lack the compute to validate at 1B+.
2. **All components are from published papers.** Our contribution is the specific combination, the LoRA-TTT finding, and the W-snapshot pyramid concept.
3. **100M context is lossy.** TTT compresses information, it doesn't store it losslessly. The "100M" claim requires significant caveats.
4. **MoE expert count is untested at scale.** 8 experts work at 50M; DeepSeek-V3 uses 256 at 671B. Optimal count at 1T is unknown.
5. **Training cost is $5B.** This is a national-scale project, not a startup endeavor.

---

## Project Structure

```
v21/
├── pretrain/
│   ├── models.py              # Core architecture (DiffAttn + MLA + TTT + MoE)
│   ├── train.py               # WikiText-103 pretraining script
│   ├── validate_refactor.py   # Quick architecture validation
│   ├── pyramid_routing_test.py    # W-snapshot routing experiment
│   ├── pyramid_tradeoff_test.py   # Storage vs accuracy tradeoff
│   ├── moe_turboquant_test.py     # MoE + quantization validation
│   ├── nexus_1t_final.py          # 1T configuration search
│   ├── scaleup_deepwater.py       # 5-point stress test suite
│   └── ttt_online_test.py         # TTT online learning measurement
├── nexus/                     # Legacy module directory
├── docs/                      # Documentation
└── README.md                  # This file
```

### Core File: `pretrain/models.py`

Contains all architecture components in a single, readable file:
- `DiffAttnMLA`: Differential Attention with Multi-head Latent Attention
- `TTTLinear`: Test-Time Training with LoRA, FP32-safe cumsum
- `MoESwiGLUFFN`: Mixture-of-Experts SwiGLU Feed-Forward
- `NexusBlock`: Scale-Aware block that auto-configures everything
- `NexusGPT`: Full model with embedding, blocks, and LM head

---

## Quick Start

### Validate Architecture (2 seconds)

```bash
python pretrain/validate_refactor.py
```

Expected output:
```
TTT LoRA rank=8, W_A: [8, 512], W_B: [512, 8]
Online learning: slope=-0.020 [OK]
Gradients: NaN=False ✅
Memory: 3864 MB
```

### Run Full Experiment Suite

```bash
# TTT online learning validation (7s)
python pretrain/ttt_online_test.py

# W-snapshot pyramid routing (0.1s)
python pretrain/pyramid_routing_test.py

# Storage vs accuracy tradeoff (24s)
python pretrain/pyramid_tradeoff_test.py

# MoE + TurboQuant validation (2s)
python pretrain/moe_turboquant_test.py

# 1T configuration search (86s)
python pretrain/nexus_1t_final.py
```

### Pretrain on WikiText-103

```bash
# Requires: pip install transformers datasets
python pretrain/train.py --model both --max-steps 5000
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA)
- GPU: RTX 4060 8GB or equivalent (for micro-benchmarks)
- Optional: `transformers`, `datasets` (for pretraining)

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{nexus2026,
  title={NEXUS: A Scale-Aware Transformer with LoRA-TTT, Differential Attention, MLA and MoE},
  author={LeBon},
  year={2026},
  url={https://github.com/lebonbruce/NEXUS-1T}
}
```

---

## Looking for Collaboration

I'm an independent researcher with limited compute resources. If you're interested in:
- **Compute sponsorship** to train NEXUS at 1B+ scale
- **Research collaboration** on TTT online learning or long-context architectures
- **Hiring** someone who understands these architectures deeply

Please reach out: lebonbruce92@gmail.com

---

## License

MIT License. Use freely, attribution appreciated.

---

*"用结构换算力，用记忆换智商" — Trade structure for compute, trade memory for intelligence.*
