"""快速验证 NEXUS 模型的单次实验"""
import torch
import numpy as np
from nexus.config import NexusConfig
from nexus.model import NEXUSTransformer
from nexus_benchmark import run_single_experiment
from sce.evaluation import compute_cl_metrics

config = NexusConfig()
config.steps_per_task = 200  # 快速验证

device = torch.device(config.device)
torch.manual_seed(42)
np.random.seed(42)

print("Running NEXUS single experiment...")
result = run_single_experiment(config, NEXUSTransformer, device)

metrics = compute_cl_metrics(result["accuracy_matrix"])
print(f"\nNEXUS Results:")
print(f"  AA:  {metrics['AA']*100:.2f}%")
print(f"  BWT: {metrics['BWT']*100:+.2f}%")
print(f"  Params: {result['final_params_trainable']:,}")
print(f"  Time: {result['duration']:.1f}s")
print(f"\nAccuracy Matrix:")
print(np.array2string(result["accuracy_matrix"], precision=3))
