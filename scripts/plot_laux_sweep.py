"""Plot lambda_aux sweep results for modular addition with MoE."""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# --- Load data ---
laux_values = [0.0, 0.001, 0.01, 0.1, 1.0]
seeds = [42, 137, 256, 512, 1024]
ckpt_dir = Path("checkpoints")

acc_data = {v: [] for v in laux_values}
epoch_data = {v: [] for v in laux_values}

for v in laux_values:
    for s in seeds:
        path = ckpt_dir / f"modadd_moe_laux{v}_s{s}" / "modadd_best.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        acc_data[v].append(ckpt["test_acc"])
        epoch_data[v].append(ckpt["epoch"])

# --- Figure setup ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
})

COLOR = "#4878A8"  # steel blue
DOT_COLOR = "#4878A8"
MEAN_COLOR = "#2C5480"

fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.6), dpi=300)
plt.subplots_adjust(wspace=0.45, left=0.12, right=0.97, bottom=0.22, top=0.90)

# X positions: place 0.0 at a pseudo-log position to the left of 0.001
# Use positions: 0, 1, 2, 3, 4 mapping to 0.0, 0.001, 0.01, 0.1, 1.0
x_positions = np.arange(len(laux_values))
x_labels = ["0", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "10$^{0}$"]

for ax_idx, (data, ylabel, title) in enumerate([
    (acc_data, "Test accuracy", "Final accuracy"),
    (epoch_data, "Epoch", "Grokking epoch"),
]):
    ax = axes[ax_idx]

    means = []
    cis = []
    for i, v in enumerate(laux_values):
        vals = np.array(data[v])
        mean = np.mean(vals)
        means.append(mean)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals))
        ci95 = 1.96 * se
        cis.append(ci95)

        # Individual seed dots with jitter
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            x_positions[i] + jitter, vals,
            s=10, color=DOT_COLOR, alpha=0.45, zorder=3, linewidths=0,
        )

    # Mean markers with error bars
    ax.errorbar(
        x_positions, means, yerr=cis,
        fmt="o", markersize=4.5, color=MEAN_COLOR,
        ecolor=MEAN_COLOR, elinewidth=0.9, capsize=2.5, capthick=0.9,
        zorder=4, markeredgewidth=0,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("$\\lambda_{\\mathrm{aux}}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="medium")

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

# Save
out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "fig_laux_sweep.png", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "fig_laux_sweep.pdf", bbox_inches="tight")
plt.close()

# Print summary
print("\n=== Lambda_aux Sweep Summary ===")
print(f"{'laux':>8s}  {'acc_mean':>8s} {'acc_std':>7s}  {'epoch_mean':>10s} {'epoch_std':>9s}")
for v in laux_values:
    a = np.array(acc_data[v])
    e = np.array(epoch_data[v])
    print(f"{v:>8.3f}  {a.mean():>8.4f} {a.std():>7.4f}  {e.mean():>10.1f} {e.std():>9.1f}")
print("\nSaved: figures/fig_laux_sweep.png, figures/fig_laux_sweep.pdf")
