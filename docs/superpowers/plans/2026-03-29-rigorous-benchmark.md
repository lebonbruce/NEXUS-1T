# Rigorous Benchmark Plan: Transformer Continual Learning

## 1. Objective
Benchmark a novel "Rocket" primitive against established Continual Learning (CL) methods in a 1M parameter Transformer on a real-world language task (WikiText-2).

## 2. Success Criteria
- The "Rocket" architecture must achieve significantly lower forgetting (Backwards Transfer Error) than standard Fine-tuning and competitive performance with EWC.
- Evaluation must use equivalent parameter budgets and training compute.
- Qualitative text generation must show logical coherence after multiple tasks.

## 3. Key Methodology
- **Dataset:** WikiText-2 (Character-level).
- **Task Split:** 
    - Era 1: First 50% of WikiText-2.
    - Era 2: Last 50% of WikiText-2 (shuffled to ensure domain shift if any).
- **Architecture:** 1M Params Transformer (D_MODEL=128, N_LAYERS=4).
- **Comparison Groups:**
    1. **Naive FT:** Standard gradient descent on Task 2.
    2. **EWC (Elastic Weight Consolidation):** Regularize weights based on Fisher Information Matrix of Task 1.
    3. **Orthogonal Rocket (Proposed):** Update weights only in the subspace orthogonal to the previous task's feature covariance.

## 4. Implementation Steps
1.  **Dataset Preparation:** Clean WikiText-2 and create train/test splits for both Eras.
2.  **Baseline Implementation:** Standard GPT training loop with checkpointing.
3.  **EWC Implementation:** Calculate Fisher matrix and add penalty term to loss.
4.  **Orthogonal Rocket Implementation:** 
    - Capture input covariance $C = \sum x x^T$ during Task 1.
    - Apply projection $P = I - C(C^T C)^{-1} C^T$ to gradients during Task 2.
5.  **Benchmarking Execution:** Run all 3 models sequentially through both Eras.
6.  **Reporting:** Compare Cross-Entropy on Task 1 test set after Task 2 training.

## 5. Risk Mitigation
- **Numerical Instability:** Orthogonal projection can be sensitive to matrix inversion; use pseudo-inverse or iterative approximation.
- **Scale:** 1M is small, but character-level language modeling is complex enough to show real failure/success modes.
