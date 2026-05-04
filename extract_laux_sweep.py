#!/usr/bin/env python
"""Extract lambda_aux sweep results from modadd MoE checkpoints."""

import torch
from pathlib import Path
from collections import defaultdict
import numpy as np

LAMBDA_AUX_VALUES = [0.0, 0.001, 0.01, 0.1, 1.0]
SEEDS = [42, 137, 256, 512, 1024]
CHECKPOINT_DIR = Path("<PATH_TO_REPO>/checkpoints")

results = defaultdict(list)

for laux in LAMBDA_AUX_VALUES:
    for seed in SEEDS:
        checkpoint_name = f"modadd_moe_laux{laux}_s{seed}"
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name / "modadd_best.pt"

        if not checkpoint_path.exists():
            print(f"WARNING: {checkpoint_path} not found")
            continue

        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            accuracy = ckpt.get('test_acc', None)
            epoch = ckpt.get('epoch', None)

            if accuracy is not None and epoch is not None:
                results[laux].append({
                    'seed': seed,
                    'accuracy': float(accuracy),
                    'epoch': int(epoch)
                })
                print(f"✓ {checkpoint_name}: acc={accuracy:.4f}, epoch={epoch}")
            else:
                print(f"WARNING: Missing keys in {checkpoint_name}")
        except Exception as e:
            print(f"ERROR loading {checkpoint_name}: {e}")

print("\n" + "="*70)
print("LAMBDA_AUX SWEEP SUMMARY")
print("="*70)
print(f"{'λ_aux':<10} {'Mean Acc':<12} {'Std Acc':<12} {'Mean Epoch':<12} {'Std Epoch':<12} {'N Seeds':<8}")
print("-"*70)

for laux in LAMBDA_AUX_VALUES:
    if laux in results and len(results[laux]) > 0:
        accs = [r['accuracy'] for r in results[laux]]
        epochs = [r['epoch'] for r in results[laux]]

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_epoch = np.mean(epochs)
        std_epoch = np.std(epochs)
        n_seeds = len(results[laux])

        print(f"{laux:<10.3f} {mean_acc:<12.4f} {std_acc:<12.4f} {mean_epoch:<12.1f} {std_epoch:<12.1f} {n_seeds:<8}")
    else:
        print(f"{laux:<10.3f} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {0:<8}")

print("="*70)
