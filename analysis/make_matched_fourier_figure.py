"""Per-neuron Fourier concentration on (a+b) mod 113 (Fig. 6, paper main).

Uses parameter-matched checkpoints aggregated across 5 seeds. No in-figure
text or arrows -- all interpretation lives in the LaTeX caption.

Width / parameter convention on modular addition (d_model = 128, h_dense = 512):
  - FFN     : h        = 512                    (32-bit baseline)
  - GLU     : h        = 340  (= floor(2/3 h_dense))   total-param matched
  - MoE     : h_E      = 128  (= h_dense / E)          total-param matched
  - MoE-GLU : h_E      = 85   (= floor(2/3 h_dense)/E) total-param matched-as-GLU
              i.e. intermediate_dim=340 split across E=4 experts.

In the codebase, MoE / MoE-GLU receive a *total* intermediate_dim and split it
evenly across experts (model/components.py: `expert_intermediate = intermediate_dim // num_experts`),
so checkpoints named "_d340" mean total=340, h_E=85 per expert.

Inputs:  checkpoints/modadd_{variant}_s{seed}/modadd_best.pt  (5 seeds)
Output:  figures/fig6_fourier_concentration.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from analysis.visualize_results import (
    P,
    SEEDS,
    COLORS,
    LABELS,
    get_all_inputs,
    load_model,
    get_neuron_activations,
    neuron_fourier_stats,
)


# Variant -> checkpoint dir template. We use the matched-width dirs for GLU
# and MoE-GLU; FFN and plain MoE are at their default (also total-matched) widths.
VARIANTS = [
    ("ffn",     "modadd_ffn_s{seed}"),           # h = 512
    ("glu",     "modadd_glu_d340_s{seed}"),      # h = 340 (2/3 * 512)
    ("moe",     "modadd_moe_s{seed}"),           # h_E = 128
    ("moe_glu", "modadd_moe_glu_d340_s{seed}"),  # h_E = 85, total = 340
]


def _ftype_from_dir(name: str) -> str:
    if "moe_glu" in name:
        return "moe_glu"
    if "moe" in name:
        return "moe"
    if "glu" in name:
        return "glu"
    return "ffn"


def gather_concentrations(template: str, inputs, a_np, b_np):
    """Aggregate per-neuron Fourier concentrations across seeds for one variant."""
    all_concs: list[float] = []
    seed_means: list[float] = []
    n_seeds = 0
    for s in SEEDS:
        ck = ROOT / "checkpoints" / template.format(seed=s) / "modadd_best.pt"
        if not ck.exists():
            print(f"  [skip] {ck.parent.name} (no best checkpoint)")
            continue
        try:
            model, _ = load_model(str(ck))
        except Exception as e:
            print(f"  [load-err] {ck.parent.name}: {e}")
            continue
        ftype = _ftype_from_dir(ck.parent.name)
        act_dict = get_neuron_activations(model, inputs, ftype)
        seed_concs: list[float] = []
        for _, act in act_dict.items():
            _, concs = neuron_fourier_stats(act, a_np, b_np, P)
            seed_concs.extend(concs)
        seed_concs = np.array(seed_concs)
        all_concs.extend(seed_concs)
        seed_means.append(float(seed_concs.mean()))
        n_seeds += 1
        print(f"  [{ck.parent.name}]  {len(seed_concs)} neurons, mean={seed_concs.mean():.3f}")
    return np.array(all_concs), np.array(seed_means), n_seeds


def main():
    inputs, _, a_vals, b_vals = get_all_inputs()
    a_np, b_np = a_vals.numpy(), b_vals.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharey=False)
    axes_flat = axes.flatten()
    summary = []
    for idx, (ftype, tmpl) in enumerate(VARIANTS):
        print(f"\n=== {LABELS[ftype]} ({tmpl.replace('_s{seed}', '')}, 5 seeds) ===")
        concs, seed_means, n_seeds = gather_concentrations(tmpl, inputs, a_np, b_np)
        ax = axes_flat[idx]
        ax.hist(concs, bins=30, color=COLORS[ftype], alpha=0.85,
                edgecolor="black", linewidth=0.6)
        mean_val = concs.mean()
        seed_std = seed_means.std()
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.8, alpha=0.85)
        ax.set_title(
            f"{LABELS[ftype]}   (mean $=$ {mean_val:.2f} $\\pm$ {seed_std:.2f})",
            fontsize=18, fontweight="bold",
        )
        # Bottom row: x-label. Left column: y-label.
        if idx >= 2:
            ax.set_xlabel("Fourier Concentration", fontsize=17, fontweight="bold")
        if idx % 2 == 0:
            ax.set_ylabel("# Neurons", fontsize=17, fontweight="bold")
        ax.tick_params(axis="both", labelsize=14)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontweight("bold")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        summary.append((LABELS[ftype], mean_val, seed_std, len(concs), n_seeds))

    fig.tight_layout()
    out = ROOT / "figures" / "fig6_fourier_concentration.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paper_out = ROOT / "paper" / "figures" / "fig6_fourier_concentration.png"
    paper_out.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(out, paper_out)
    print(f"\n-> wrote {out.relative_to(ROOT)}")
    print(f"-> copied to {paper_out.relative_to(ROOT)}")

    print("\nSUMMARY")
    print("-" * 72)
    for label, mean, sd, n_neurons, n_seeds in summary:
        print(f"  {label:8s}  conc = {mean:.3f} +/- {sd:.3f} (across {n_seeds} seed means)   "
              f"n_neurons={n_neurons}")


if __name__ == "__main__":
    main()
