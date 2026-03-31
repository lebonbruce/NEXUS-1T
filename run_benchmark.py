"""
SCE Benchmark: 一键运行持续学习对比实验。

用法: python run_benchmark.py

输出: 5个方法在5个算法推理任务上的标准CL指标对比表。
"""
import sys
import os
import numpy as np

# 确保 sce 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sce.config import ExperimentConfig
from sce.runner import run_all_experiments
from sce.tasks import TASK_NAMES


def print_results_table(results: dict, config: ExperimentConfig):
    """格式化打印结果对比表。"""
    print("\n")
    print("=" * 80)
    print("STRUCTURAL COGNITIVE ENGINE — RIGOROUS BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Tasks: {config.num_tasks} | Steps/task: {config.steps_per_task} | "
          f"Runs: {config.num_runs} | Vocab: {config.vocab_size} | "
          f"SeqLen: {config.seq_len}")
    print("-" * 80)

    # 表头
    header = (f"{'Method':15s} | {'AA (↑)':12s} | {'BWT (↑)':12s} | "
              f"{'Params (T)':>10s} | {'Params (All)':>12s} | {'Time':>6s}")
    print(header)
    print("-" * 80)

    for name, res in results.items():
        m = res["metrics_mean"]
        s = res["metrics_std"]
        aa_str = f"{m['AA']*100:.1f}±{s['AA']*100:.1f}%"
        bwt_str = f"{m['BWT']*100:+.1f}±{s['BWT']*100:.1f}%"
        pt = f"{res['params_trainable']/1e3:.0f}K"
        pa = f"{res['params_total']/1e3:.0f}K"
        t = f"{res['duration_mean']:.0f}s"
        print(f"{name:15s} | {aa_str:12s} | {bwt_str:12s} | "
              f"{pt:>10s} | {pa:>12s} | {t:>6s}")

    print("-" * 80)
    print("AA = Average Accuracy (higher is better)")
    print("BWT = Backward Transfer (higher is better, negative = forgetting)")
    print("Params (T) = trainable at end, Params (All) = total including frozen")
    print("=" * 80)

    # 打印每个方法训练完所有任务后的per-task accuracy
    print("\nPer-Task Accuracy (after all tasks trained):")
    print("-" * 80)
    task_header = f"{'Method':15s} |"
    for tn in TASK_NAMES:
        task_header += f" {tn[:10]:>10s}"
    print(task_header)
    print("-" * 80)

    for name, res in results.items():
        # 取第一次run的矩阵
        mat = res["acc_matrices"][0]
        row = f"{name:15s} |"
        for tid in range(config.num_tasks):
            row += f" {mat[tid, -1]*100:9.1f}%"
        print(row)

    print("-" * 80)

    # 诚实结论
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # 找出最好的方法
    best_aa_name = max(results.keys(), key=lambda n: results[n]["metrics_mean"]["AA"])
    best_bwt_name = max(results.keys(), key=lambda n: results[n]["metrics_mean"]["BWT"])

    print(f"Best Average Accuracy: {best_aa_name} "
          f"({results[best_aa_name]['metrics_mean']['AA']*100:.1f}%)")
    print(f"Best Anti-Forgetting:  {best_bwt_name} "
          f"({results[best_bwt_name]['metrics_mean']['BWT']*100:+.1f}%)")

    # 与Naive的对比
    naive_aa = results["Naive"]["metrics_mean"]["AA"]
    sce_aa = results["SCE (ours)"]["metrics_mean"]["AA"]
    naive_bwt = results["Naive"]["metrics_mean"]["BWT"]
    sce_bwt = results["SCE (ours)"]["metrics_mean"]["BWT"]

    print(f"\nSCE vs Naive:")
    print(f"  AA improvement: {(sce_aa - naive_aa)*100:+.1f} percentage points")
    print(f"  BWT improvement: {(sce_bwt - naive_bwt)*100:+.1f} percentage points")

    sce_params = results["SCE (ours)"]["params_total"]
    prog_params = results["Progressive"]["params_total"]
    print(f"\nSCE vs Progressive (parameter efficiency):")
    print(f"  SCE total params:         {sce_params:,}")
    print(f"  Progressive total params: {prog_params:,}")
    print(f"  Ratio: SCE uses {sce_params/prog_params*100:.0f}% of Progressive's params")


def main():
    config = ExperimentConfig()

    print("=" * 80)
    print("SCE BENCHMARK — 用结构换算力，用记忆换智商")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Config: {config.num_tasks} tasks × {config.steps_per_task} steps, "
          f"{config.num_runs} runs")
    print("Methods: Naive, EWC, Replay, Progressive, SCE")
    print("=" * 80)

    results = run_all_experiments(config)
    print_results_table(results, config)


if __name__ == "__main__":
    main()
