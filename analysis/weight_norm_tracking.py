"""
Weight norm tracking during grokking on modular addition.

Loads epoch checkpoints and measures L2 norm of attention and FFN weights
over training. Tests whether MoE's aux loss constrains weight growth
during the memorization phase, providing a mechanistic explanation
for why it accelerates grokking.
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ".")
sys.path.insert(0, "model")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

from model import OneLayerTransformer

DEVICE = "cpu"
SEEDS = [42, 137, 256, 512, 1024]
EPOCH_CHECKPOINTS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]

COLORS = {
    'ffn': '#1f77b4', 'glu': '#ff7f0e',
    'moe': '#2ca02c', 'moe_glu': '#d62728',
}
LABELS = {
    'ffn': 'FFN', 'glu': 'GLU',
    'moe': 'MoE', 'moe_glu': 'MoE-GLU',
}


def compute_norms(state_dict):
    """Compute L2 norms of attention and FFN weight groups."""
    attn_norm = 0.0
    ffn_norm = 0.0
    total_norm = 0.0

    for name, param in state_dict.items():
        n = param.float().norm().item() ** 2
        total_norm += n
        if 'atn' in name:
            attn_norm += n
        elif 'ffn' in name:
            ffn_norm += n

    return {
        'attn': np.sqrt(attn_norm),
        'ffn': np.sqrt(ffn_norm),
        'total': np.sqrt(total_norm),
    }


def plot_weight_norms():
    """Plot weight norms over training for all 4 variants."""
    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    norm_types = ['attn', 'ffn', 'total']
    norm_labels = ['Attention Weights', 'FFN Weights', 'Total Weights']

    for ftype in ftypes:
        # Collect norms across seeds and epochs
        all_norms = {nt: np.zeros((len(SEEDS), len(EPOCH_CHECKPOINTS))) for nt in norm_types}

        for si, seed in enumerate(SEEDS):
            for ei, ep in enumerate(EPOCH_CHECKPOINTS):
                path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_epoch{ep}.pt"
                if os.path.exists(path):
                    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
                    norms = compute_norms(ckpt['model_state_dict'])
                    for nt in norm_types:
                        all_norms[nt][si, ei] = norms[nt]

        for ni, (nt, label) in enumerate(zip(norm_types, norm_labels)):
            ax = axes[ni]
            # Only use seeds that have data
            valid = all_norms[nt].sum(axis=1) > 0
            if valid.sum() == 0:
                continue
            data = all_norms[nt][valid]
            mean = data.mean(axis=0)
            std = data.std(axis=0)

            ax.plot(EPOCH_CHECKPOINTS, mean, 'o-', color=COLORS[ftype],
                    label=LABELS[ftype], linewidth=2, markersize=4)
            ax.fill_between(EPOCH_CHECKPOINTS, mean - std, mean + std,
                            color=COLORS[ftype], alpha=0.15)

        print(f"  {ftype} done")

    for ni, label in enumerate(norm_labels):
        axes[ni].set_xlabel("Epoch")
        axes[ni].set_ylabel("L2 Norm")
        axes[ni].set_title(label)
        axes[ni].legend()
        axes[ni].grid(True, alpha=0.2)

    fig.suptitle("Weight Norm Growth During Grokking (Modular Addition, no norm, 5 seeds)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_weight_norms.png")
    print("Saved fig_weight_norms.png")


def print_summary():
    """Print weight norm at start vs end of training."""
    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']

    print("=== Weight Norm Summary (epoch 5000 vs 40000) ===\n")

    for ftype in ftypes:
        early_norms = {'attn': [], 'ffn': [], 'total': []}
        late_norms = {'attn': [], 'ffn': [], 'total': []}

        for seed in SEEDS:
            for ep, store in [(5000, early_norms), (40000, late_norms)]:
                path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_epoch{ep}.pt"
                if os.path.exists(path):
                    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
                    norms = compute_norms(ckpt['model_state_dict'])
                    for nt in norms:
                        store[nt].append(norms[nt])

        print(f"{LABELS[ftype]}:")
        for nt in ['attn', 'ffn', 'total']:
            if early_norms[nt] and late_norms[nt]:
                e = np.array(early_norms[nt])
                l = np.array(late_norms[nt])
                ratio = l.mean() / e.mean()
                print(f"  {nt:>6}: ep5k={e.mean():.2f}+/-{e.std():.2f}  "
                      f"ep40k={l.mean():.2f}+/-{l.std():.2f}  "
                      f"ratio={ratio:.2f}x")
        print()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    print("Weight Norm Tracking During Grokking\n")
    print_summary()
    print("Generating figure...")
    plot_weight_norms()
    print("\nDone!")
