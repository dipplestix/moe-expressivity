"""Generate paper Figure 1 (fig1_headline) with the post-feedback layout.

Values are taken directly from the existing Figure 1 / Tab.~\\ref{tab:random-routing}
on add-7 (no-FFN ablation, digit-only metric, 5 seeds each). They are not
recomputed here — see analysis/eval_unified_routing.py if you want to
re-derive them.

Layout follows the post-review feedback:
  - One dashed chance line at 10% labeled "chance" (the only dashed line).
  - Three subtle background bands grouping conditions by regime
    (dense / no routing, reduced active capacity, top-1 sparse partitioning).
  - Two-line x-tick labels, no rotation.
  - Bracket over the last two conditions with a claim-positive annotation:
    "no detectable learned-routing effect / Delta = +3.6 pp, p = 0.60".
  - Lighter typography on axes; numeric labels emphasized only on the
    learned-vs-random comparison.

Outputs (overwrite):
  figures/fig1_headline.png
  figures/fig1_headline.pdf
  paper/figures/fig1_headline.png
  paper/figures/fig1_headline.pdf
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


# (label_top, label_bot, mean, std, color)
# Means and stds match Tab. random-routing (digit-only metric, average over o0..o3).
ROWS = [
    ("Dense FFN",  "baseline",      11.9, 3.6, "#9AA0A6"),
    ("GLU",        "matched",        9.7, 2.4, "#9AA0A6"),
    ("Narrow FFN", "h = 64",        23.7, 5.3, "#5B8FCB"),
    ("MoE top-2",  "2 of 4 active", 18.7, 6.9, "#5B8FCB"),
    ("MoE top-1",  "learned",       44.3, 12.5, "#4FA987"),
    ("MoE top-1",  "random",        49.1, 9.5, "#4FA987"),
]

GROUP_BANDS = [
    (0, 1, "dense / no routing",        "#EFEFEF"),
    (2, 3, "reduced active capacity",   "#E3ECF7"),
    (4, 5, "top-1 sparse partitioning", "#E2F0EA"),
]

P_VALUE_TEXT = r"$\Delta = +4.8$ pp,  $p = 0.56$"


def plot(rows, save_paths):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    xs = np.arange(len(rows))

    for lo, hi, _label, fill in GROUP_BANDS:
        ax.axvspan(lo - 0.45, hi + 0.45, color=fill, alpha=0.55, zorder=0,
                   linewidth=0)

    ax.axhline(10.0, linestyle="--", color="#888888", linewidth=1.4, zorder=1)
    ax.text(len(rows) - 0.5, 10.0 + 1.6, "chance",
            color="#555555", fontsize=14, ha="right", va="bottom")

    for i, (top, bot, mean, std, color) in enumerate(rows):
        ax.errorbar(i, mean, yerr=std, fmt="o", color=color, markersize=14,
                    markeredgecolor="white", markeredgewidth=1.4,
                    elinewidth=2.0, capsize=6, capthick=1.6, zorder=4)
        emphasis = i >= len(rows) - 2
        ax.text(i + 0.22, mean, f"{mean:.1f}%",
                fontsize=15,
                fontweight=("semibold" if emphasis else "normal"),
                color="#1a1a1a", va="center", ha="left", zorder=5)

    last_lo, last_hi = len(rows) - 2, len(rows) - 1
    bracket_y = max(rows[last_lo][2] + rows[last_lo][3],
                    rows[last_hi][2] + rows[last_hi][3]) + 3.5
    ax.plot([last_lo, last_lo, last_hi, last_hi],
            [bracket_y - 1.6, bracket_y, bracket_y, bracket_y - 1.6],
            color="#333333", linewidth=1.4, zorder=3)
    ax.text((last_lo + last_hi) / 2, bracket_y + 8.0,
            "no detectable learned-routing effect",
            ha="center", va="bottom", fontsize=14, color="#1a1a1a")
    ax.text((last_lo + last_hi) / 2, bracket_y + 1.8,
            P_VALUE_TEXT,
            ha="center", va="bottom", fontsize=13, color="#333333")

    band_y = -22.0
    for lo, hi, label, _fill in GROUP_BANDS:
        ax.annotate("",
                    xy=(lo - 0.3, band_y), xytext=(hi + 0.3, band_y),
                    arrowprops=dict(arrowstyle="-", color="#999999",
                                    linewidth=1.2),
                    annotation_clip=False)
        ax.text((lo + hi) / 2, band_y - 2.5, label,
                ha="center", va="top", fontsize=13, color="#555555",
                fontstyle="italic", clip_on=False)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r[0]}\n{r[1]}" for r in rows],
                       fontsize=15, linespacing=1.3)
    ax.tick_params(axis="x", which="both", length=0, pad=8)

    ax.set_ylim(0, 92)
    ax.set_yticks(np.arange(0, 71, 10))
    ax.set_ylabel("No-FFN accuracy (%)", fontsize=17, fontweight="normal",
                  color="#1a1a1a", labelpad=10)
    ax.tick_params(axis="y", colors="#333333", labelsize=14)

    ax.set_xlim(-0.6, len(rows) - 0.4)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#888888")
        ax.spines[sp].set_linewidth(0.9)

    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


def main():
    save_paths = [
        ROOT / "figures" / "fig1_headline.png",
        ROOT / "figures" / "fig1_headline.pdf",
    ]
    plot(ROWS, save_paths)

    paper_dir = ROOT / "paper" / "figures"
    for src in save_paths:
        dst = paper_dir / src.name
        shutil.copy2(src, dst)
        print(f"  mirrored -> {dst}")


if __name__ == "__main__":
    main()
