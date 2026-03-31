# NEXUS v2 认知架构 — 深度优化交接 Prompt

## 项目概述
你正在接手一个名为 **NEXUS (Neural EXtensible Unified System)** 的认知架构项目。目标是融合所有前沿 Transformer 架构优化技术，打造一台综合的、全新的 T 架构，效率拉满，各项指标全面超越传统 T 架构。

项目位于 `v21/` 目录。运行在 **Windows + CUDA (RTX GPU)** 环境。

---

## 当前架构组件（10 大技术）

### 核心注意力层
1. **Differential Attention**（Microsoft）— `nexus/diff_attention.py`
   - 实现：⭐⭐⭐⭐ 高度忠实。lambda 计算、SubLN、(1-lambda_init) 缩放均与官方一致
   - 融合：内嵌了 DeepSeek-V3 MLA 低秩 KV 压缩 + Apple KV Cache Sharing

### 推理时学习
2. **TTT-Linear**（Stanford）— `nexus/ttt_layer.py`
   - 实现：⭐⭐⭐⭐ 参考官方 `ttt_source.py` 的 Dual Form
   - 每个 mini-batch 通过自监督重构任务更新内部权重 W

3. **Neural Memory (Titans)**（Google Research）— `nexus/neural_memory.py`
   - 实现：⭐⭐⭐ 使用 `torch.func.vmap + grad` 实现 per-sample 梯度
   - **已修复**：eval 模式下跳过 store_memories 以兼容 EGGROLL

### 记忆系统
4. **MSA (Memory Sparse Attention)**（EverMind, arxiv:2603.23516）— `nexus/msa_memory.py` ← **NEW**
   - 1亿 token 级别的文档级语义路由 + 任务级 KV snapshot
   - 在 `on_task_end` 时自动存储，推理时通过语义投影检索 top-k

### KV Cache 压缩
5. **TurboQuant V3**（Google, ICLR 2026, arxiv:2504.19874）— `nexus/int8_kv_cache.py` ← **NEW (完全重写)**
   - **算法**：PolarQuant 随机正交旋转 + Lloyd-Max 最优标量量化
   - **忠实参考**：tonbistudio/turboquant-pytorch（603⭐，社区 V3 最佳实践）
   - **关键发现**：QJL (论文 Stage 2) 在 KV cache 中有害（6个独立团队确认），已正确去掉
   - **配置**：非对称 K4/V2（Keys 4-bit, Values 2-bit），**5.7x 压缩**，CosSim=0.995

### 前向训练
6. **EGGROLL**（NVIDIA, arxiv:2511.16652）— `nexus/eggroll_trainer.py` ← **NEW**
   - 低秩扰动 + 反义对采样的进化策略
   - **诚实评估**：在 1.6M 参数 toy 模型上完全不收敛（AA=4.7%）
   - 论文优势在 10B+ 参数时才显现

### 其他组件
7. **FFT Token Mixing**（FNet, Google）— 内嵌于 `nexus/model.py`
8. **SwiGLU FFN + 细胞分裂/合并 MoE** — `nexus/moe_ffn.py`
9. **EWC (Elastic Weight Consolidation)** — `nexus/ewc.py`
10. **Experience Replay** — 内嵌于 `nexus/model.py`

---

## 最新 Benchmark 结果（nexus_v2_benchmark.py 输出）

### TurboQuant V3 压缩质量
```
K4/V2 (推荐, 平均 3-bit):
  Keys:   MSE=0.009247 | CosSim=0.995329 | 3.9x 压缩
  Values: MSE=0.116258 | CosSim=0.940304 | 7.5x 压缩
  平均压缩比: 5.7x
```

### 持续学习对比
```
Method                 |     AA |     BWT |   Params |   Time |  GPU MB
Naive                  |  22.1% |  -60.6% |    812K |    24s |   66.2
NEXUS v2 (backward)    |  52.9% |   -4.9% |   1610K |  1064s |  592.2
NEXUS v2 (EGGROLL)     |   4.7% |   -5.1% |   1610K |   771s |  594.9
```

### Per-Task 最终准确率
```
                       | LinearMap | Reversal | CumSum | Sort | Parity
Naive                  |     0.8%  |    2.7%  |   2.0% |  5.0%|  100.0%
NEXUS v2 (backward)    |    99.5%  |   87.3%  |  15.2% | 16.4%|   46.0%
NEXUS v2 (EGGROLL)     |     5.2%  |    1.5%  |   1.7% |  1.8%|   13.1%
```

---

## 已知问题清单（需要本轮深度排查）

### 🔴 致命级
1. **训练速度：NEXUS 300步 ≈ 1064s，Naive ≈ 24s（44倍慢）**
   - NeuralMemory 的 vmap+grad 是主要瓶颈
   - TTT 的内部 mini-batch 循环
   - 每步 Replay 双倍前向+反向
   - **目标**：在不损失功能的前提下降到 10 倍以内

### 🟡 性能级
2. **CumulativeSum (15.2%) 和 Sort (16.4%) 学不好**
   - seq_len=16 太短，TTT/NeuralMemory 设计用于长序列
   - 300步可能不够
3. **ParityEncode 回退（从以前的 82.8% 降到 46.0%）**
   - 可能是 MSA 集成引入了干扰
4. **EGGROLL 完全无法收敛**
   - 在 toy 模型上是预期行为，但需要明确：在什么 scale 下值得启用？

### 🟢 架构级
5. **MoE 过度合并**
   - 深拷贝初始化 → 权重本身高度相似 → 功能相似度 > 0.99 → 全部合并
   - 合并后只剩 1 个 expert，MoE 退化为普通 FFN
6. **参数膨胀**
   - 5 任务后总参数 4.6M（可训练 1.6M）
   - 需要 LoRA-style 低秩偏移替代物理冻结

---

## 代码结构

```
v21/
├── nexus/                          # NEXUS 核心架构
│   ├── config.py                   # NexusConfig（继承 ExperimentConfig）
│   ├── model.py                    # NEXUSTransformer（主模型 + CL 接口）
│   ├── diff_attention.py           # DiffAttn + MLA + KV Sharing
│   ├── ttt_layer.py                # TTT-Linear (Dual Form)
│   ├── neural_memory.py            # Titans Neural Memory (vmap+grad)
│   ├── moe_ffn.py                  # 细胞分裂/合并 SwiGLU FFN
│   ├── msa_memory.py               # MSA 1亿token 记忆稀疏注意力 ← NEW
│   ├── int8_kv_cache.py            # TurboQuant V3 (PolarQuant+Lloyd-Max) ← REWRITE
│   ├── eggroll_trainer.py          # EGGROLL 进化策略训练器 ← NEW
│   ├── ewc.py                      # Elastic Weight Consolidation
│   └── __init__.py
├── sce/                            # 基础框架（任务生成 + 评估 + 基线模型）
│   ├── config.py                   # ExperimentConfig
│   ├── tasks.py                    # 5个算法任务生成器
│   ├── evaluation.py               # AA, BWT 等指标
│   └── models/                     # Naive/EWC/Replay/Progressive/SCE 基线
├── nexus_v2_benchmark.py           # v2 三大技术对比 Benchmark ← NEW
├── nexus_benchmark.py              # v1 原始 Benchmark
├── ttt_benchmark.py                # TTT 单独 Benchmark
└── ttt_source.py                   # Stanford TTT 官方源码（参考用）
```

---

## 本轮任务：深度排查 + 训练加速 + 架构验证

### Phase 1：深度排查（驻足总结）
请你：
1. **逐个读取 `nexus/` 下的每个 .py 文件**（config.py, model.py, diff_attention.py, ttt_layer.py, neural_memory.py, moe_ffn.py, msa_memory.py, int8_kv_cache.py, eggroll_trainer.py, ewc.py）
2. **对每个组件做深度 code review**：
   - 实现是否忠实于论文？
   - 是否有隐藏的 bug 或性能陷阱？
   - 组件之间是否有不必要的耦合或冲突？
   - 每个组件对最终性能的贡献是多少？（需要 ablation 思路）
3. **输出一份诚实的架构审计报告**：优势/劣势/风险/建议

### Phase 2：训练速度优化
4. **Profile 训练瓶颈**：哪些组件消耗最多时间？
5. **实施优化**（不损失功能）：
   - NeuralMemory 的 vmap+grad 能否替换为更快的实现？
   - TTT 在短序列时是否值得启用？
   - Replay 频率是否需要降低？
   - 哪些组件可以条件启用（根据序列长度、训练阶段等）？

### Phase 3：架构验证 + Scale-up
6. **精简 Benchmark**：设计一个 < 5 分钟完成的快速验证
7. **验证架构鲁棒性**：在不同配置下是否稳定
8. **Scale-up 路线图**：当前架构如何从 1.6M → 10M → 100M 参数扩展

### 核心原则
- **不做玩具**：每个决定都要考虑 scale-up
- **诚实**：如果某个组件没有实际贡献，直说
- **效率**：训练速度是第一优先级
- **始终用中文交流**

---

## 环境信息
- OS: Windows + CUDA (RTX GPU)
- Python 3.12 + PyTorch (CUDA)
- 项目路径: `v21/`
- 运行: `python nexus_v2_benchmark.py`（完整 benchmark，约 30 分钟）
