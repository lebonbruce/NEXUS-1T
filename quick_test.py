"""快速验证修复后的SCE（单次运行）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from sce.config import ExperimentConfig
from sce.runner import run_single_experiment
from sce.models.sce_model import SCETransformer
from sce.evaluation import compute_cl_metrics
from sce.tasks import TASK_NAMES

torch.manual_seed(42)
np.random.seed(42)

c = ExperimentConfig()
device = torch.device(c.device)
r = run_single_experiment(c, SCETransformer, device)
m = compute_cl_metrics(r["accuracy_matrix"])

print("\n\n" + "="*50)
print("SCE FIXED - SINGLE RUN RESULTS")
print("="*50)
print(f"AA  = {m['AA']*100:.1f}%")
print(f"BWT = {m['BWT']*100:+.1f}%")
print(f"Params: {r['final_params_trainable']:,} trainable / {r['final_params_total']:,} total")
print(f"Time: {r['duration']:.0f}s")
print("\nPer-task (after all tasks):")
for i in range(5):
    print(f"  {TASK_NAMES[i]:15s}: {r['accuracy_matrix'][i,-1]*100:.1f}%")
