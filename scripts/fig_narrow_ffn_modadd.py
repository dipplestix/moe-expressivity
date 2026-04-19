"""Generate publication figure: narrow dense FFN vs normal dense FFN vs MoE on modular addition."""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────────
seeds = [42, 137, 256, 512, 1024]
configs = {
    "Dense FFN\n(full)": ("modadd_ffn_s", "#3572B0"),
    "Dense FFN\n(narrow)": ("modadd_ffn_narrow_s", "#7BAFD4"),
    "MoE": ("modadd_moe_s", "#E8773A"),
}

data = {}  # name -> {"acc": [], "epoch": []}
for name, (prefix, _) in configs.items():
    accs, epochs = [], []
    for s in seeds:
        ckpt = torch.load(
            f"checkpoints/{prefix}{s}/modadd_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        accs.append(ckpt["test_acc"] * 100)
        epochs.append(ckpt["epoch"])
    data[name] = {"acc": np.array(accs), "epoch": np.array(epochs)}

# ── Figure setup ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.8), gridspec_kw={"wspace": 0.55})

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
})

names_display = ["Dense\n(full)", "Dense\n(narrow)", "MoE"]
names = list(configs.keys())
colors = [configs[n][1] for n in names]
x = np.arange(len(names))
bar_width = 0.55

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)

# ── Panel A: Final accuracy ───────────────────────────────────────────────────
ax = axes[0]
means = [data[n]["acc"].mean() for n in names]
bars = ax.bar(x, means, bar_width, color=colors, edgecolor="white", linewidth=0.4, zorder=2)

# Individual seeds as dots
rng = np.random.RandomState(0)
for i, name in enumerate(names):
    vals = data[name]["acc"]
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(
        x[i] + jitter, vals, s=12, color="white", edgecolors=colors[i],
        linewidths=0.7, zorder=3, clip_on=False,
    )

ax.set_xticks(x)
ax.set_xticklabels(names_display, fontsize=6.5)
ax.set_ylabel("Test accuracy (%)", fontsize=7.5)
ax.set_ylim(0, 109)
ax.set_title("(a) Final accuracy", fontsize=8, pad=4)

# Annotate solve rates
for i, name in enumerate(names):
    n_solved = (data[name]["acc"] > 99).sum()
    ax.text(x[i], means[i] + 4, f"{n_solved}/5", ha="center", va="bottom", fontsize=6, fontweight="bold")

# ── Panel B: Grokking epoch (solved seeds only) ──────────────────────────────
ax = axes[1]

# Only include seeds that solved (>99% accuracy)
solved_epochs = []
for name in names:
    mask = data[name]["acc"] > 99
    solved_epochs.append(data[name]["epoch"][mask] / 1000)  # in thousands

means_e = [e.mean() if len(e) > 0 else 0 for e in solved_epochs]
bars = ax.bar(x, means_e, bar_width, color=colors, edgecolor="white", linewidth=0.4, zorder=2)

for i, name in enumerate(names):
    vals = solved_epochs[i]
    if len(vals) == 0:
        continue
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(
        x[i] + jitter, vals, s=12, color="white", edgecolors=colors[i],
        linewidths=0.7, zorder=3, clip_on=False,
    )

ax.set_xticks(x)
ax.set_xticklabels(names_display, fontsize=6.5)
ax.set_ylabel("Grokking epoch (k)", fontsize=7.5)
ax.set_title("(b) Epochs to grok", fontsize=8, pad=4)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

# ── Save ───────────────────────────────────────────────────────────────────────
fig.tight_layout(pad=0.4)
for fmt in ["png", "pdf"]:
    path = f"figures/fig_narrow_ffn_modadd.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=(fmt == "pdf"))
    print(f"Saved {path}")

plt.close()
