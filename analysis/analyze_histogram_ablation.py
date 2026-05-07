"""
H1 component ablation for histogram counting across all 4 variants x 5 seeds.
Zero attention output vs zero FFN output, measure test accuracy.

Mirrors analyze_modadd_ablation.py but uses histogram checkpoints.
"""

import sys
import os
os.chdir("<PATH_TO_REPO>")
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "model")
from model import OneLayerTransformer
from data.histogram import HistogramDataset

LABELS = {"ffn": "FFN", "glu": "GLU", "moe": "MoE", "moe_glu": "MoE-GLU"}

DEVICE = "cpu"

MODEL_KEYS = {
    "model_dim", "num_heads", "ffn_type", "dropout", "vocab_size",
    "max_seq_len", "use_norm", "is_causal", "tie_embeddings",
    "activation", "intermediate_dim", "num_experts", "top_k",
}


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    config = {k: v for k, v in ckpt["config"].items() if k in MODEL_KEYS}
    num_classes = ckpt["config"]["num_classes"]

    model = OneLayerTransformer(**config).to(DEVICE)
    model.unembed = nn.Linear(config["model_dim"], num_classes, bias=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, num_classes


def evaluate(model, inputs, targets):
    with torch.no_grad():
        logits = model(inputs)
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def ablation_study(model, inputs, targets):
    normal_acc = evaluate(model, inputs, targets)

    orig_attn = model.atn.forward
    model.atn.forward = lambda *args, **kwargs: torch.zeros_like(orig_attn(*args, **kwargs))
    no_attn_acc = evaluate(model, inputs, targets)
    model.atn.forward = orig_attn

    orig_ffn = model.ffn.forward
    model.ffn.forward = lambda *args, **kwargs: torch.zeros_like(orig_ffn(*args, **kwargs))
    no_ffn_acc = evaluate(model, inputs, targets)
    model.ffn.forward = orig_ffn

    return normal_acc, no_attn_acc, no_ffn_acc


if __name__ == "__main__":
    dataset = HistogramDataset(T=32, L=10, n_train=10000, n_test=3000, seed=42, device=DEVICE)
    inputs = dataset.test_inputs
    targets = dataset.test_targets
    print(f"Evaluating on {len(inputs)} histogram test examples\n")

    ftypes = ["ffn", "glu", "moe", "moe_glu"]
    seeds = [42, 137, 256, 512, 1024]

    all_results = {}

    for ftype in ftypes:
        print(f"{'='*50}")
        print(f"  {ftype.upper()}")
        print(f"{'='*50}")

        normals, no_attns, no_ffns = [], [], []

        for seed in seeds:
            path = f"checkpoints/hist_{ftype}_s{seed}/hist_best.pt"
            if not os.path.exists(path):
                print(f"  s{seed}: MISSING {path}")
                continue
            model, ckpt, _ = load_model(path)
            normal, no_attn, no_ffn = ablation_study(model, inputs, targets)
            normals.append(normal)
            no_attns.append(no_attn)
            no_ffns.append(no_ffn)
            print(f"  s{seed}: normal={normal:.4f} no_attn={no_attn:.4f} no_ffn={no_ffn:.4f}")

        if not normals:
            continue

        n = np.array(normals)
        a = np.array(no_attns)
        f = np.array(no_ffns)
        print(f"\n  Aggregated:")
        print(f"    Normal:  {n.mean():.4f} +/- {n.std():.4f}")
        print(f"    No attn: {a.mean():.4f} +/- {a.std():.4f}")
        print(f"    No FFN:  {f.mean():.4f} +/- {f.std():.4f}")
        print()

        all_results[ftype] = {
            "normal": (float(n.mean()), float(n.std())),
            "no_attn": (float(a.mean()), float(a.std())),
            "no_ffn": (float(f.mean()), float(f.std())),
        }

    print(f"\n{'='*50}")
    print(f"  SUMMARY: Component Ablation on Histogram")
    print(f"{'='*50}")
    print(f"{'Variant':<12} {'Normal':>10} {'No Attn':>14} {'No FFN':>14}")
    print(f"{'-'*50}")
    for ftype in ftypes:
        if ftype not in all_results:
            continue
        r = all_results[ftype]
        print(f"{ftype:<12} {r['normal'][0]:>8.4f}   "
              f"{r['no_attn'][0]:>7.4f}+/-{r['no_attn'][1]:.4f} "
              f"{r['no_ffn'][0]:>7.4f}+/-{r['no_ffn'][1]:.4f}")

    os.makedirs("figures", exist_ok=True)
    np.savez("figures/hist_ablation_results.npz", results=all_results)
    print(f"\nSaved to figures/hist_ablation_results.npz")

    plot_ftypes = [f for f in ftypes if f in all_results]
    if not plot_ftypes:
        print("No results to plot.")
        sys.exit(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plot_ftypes))
    width = 0.25

    normal_means = [all_results[f]["normal"][0] for f in plot_ftypes]
    no_attn_means = [all_results[f]["no_attn"][0] for f in plot_ftypes]
    no_attn_stds = [all_results[f]["no_attn"][1] for f in plot_ftypes]
    no_ffn_means = [all_results[f]["no_ffn"][0] for f in plot_ftypes]
    no_ffn_stds = [all_results[f]["no_ffn"][1] for f in plot_ftypes]

    ax.bar(x - width, no_attn_means, width, label="No Attention",
           color="#ff9999", edgecolor="black", linewidth=0.5,
           yerr=no_attn_stds, capsize=3)
    ax.bar(x, no_ffn_means, width, label="No FFN",
           color="#9999ff", edgecolor="black", linewidth=0.5,
           yerr=no_ffn_stds, capsize=3)
    ax.bar(x + width, normal_means, width, label="Normal",
           color="#99cc99", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in plot_ftypes])
    ax.set_ylabel("Accuracy")
    ax.set_title("Component Ablation on Histogram (5 seeds)")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig("figures/fig_hist_ablation.png", dpi=150)
    print("Saved figures/fig_hist_ablation.png")
