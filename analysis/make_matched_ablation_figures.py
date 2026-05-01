"""Generate clean component-ablation grouped-bar figures using parameter-matched widths.

Matched widths:
  - GLU and MoE-GLU on add-7:    h = 170  (h_dense = 256, ratio 2/3)
  - GLU and MoE-GLU on modadd:   h = 340  (h_dense = 512, ratio 2/3)
  - GLU and MoE-GLU on hist:     h = 340  (same as modadd)
  - Plain MoE: h_E = h_dense / E (default; already total-matched)
  - Dense FFN: full width (default)

Activations used (paper convention):
  - add-7  : SiLU
  - modadd : GELU
  - hist   : GELU

Outputs (overwrites existing pre-rendered PNGs):
  paper/images/add-7/MoE_redistributes_computation/ablation_+7_5_seeds.png
  paper/images/mod-add/MoE_redistributes_computation/ablation_modular_addition_5_seeds.png
  paper/images/hist/ablation_histogram_5_seeds.png  (new)

No in-figure arrows or text annotations - all interpretation goes in the LaTeX caption.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from analysis.analyze_activation_symmetry import (
    SEEDS,
    load_model,
    build_add7_data,
    build_modadd_data,
    build_hist_data,
    ablate_modadd,
    ablate_hist,
)


def ablate_add7(m, x, y, out_start):
    """Per-token accuracy on add-7 output digits and EOS, excluding the trailing PAD position
    (target[-1] is the PAD-prediction, not part of the spec)."""
    sl = slice(out_start, -1)
    with torch.no_grad():
        logits = m(x)
    normal = (logits.argmax(-1)[:, sl] == y[:, sl]).float().mean().item()
    orig = m.atn.forward
    m.atn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        l = m(x)
    no_attn = (l.argmax(-1)[:, sl] == y[:, sl]).float().mean().item()
    m.atn.forward = orig
    orig = m.ffn.forward
    m.ffn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        l = m(x)
    no_ffn = (l.argmax(-1)[:, sl] == y[:, sl]).float().mean().item()
    m.ffn.forward = orig
    return normal, no_attn, no_ffn


VARIANT_ORDER = ["FFN", "GLU", "MoE", "MoE-GLU"]
COLOR_NORMAL  = "#7AB57A"   # green
COLOR_NO_ATTN = "#E89C9C"   # red
COLOR_NO_FFN  = "#9B9BD8"   # purple/blue


def _gather(task_dir_templates, ckpt_filenames, ablate_fn, x, y, extra=None):
    """Run ablations for a list of (label, dir_template) and return dict of arrays."""
    out = {}
    for label, tmpl in task_dir_templates:
        ns, nas, nfs = [], [], []
        for s in SEEDS:
            d = ROOT / "checkpoints" / tmpl.format(seed=s)
            ck = None
            for fn in ckpt_filenames:
                if (d / fn).exists():
                    ck = d / fn
                    break
            if ck is None:
                continue
            try:
                m, _, _ = load_model(ck)
            except Exception as e:
                print(f"  [load-err] {d.name}: {e}")
                continue
            if extra is not None:
                n, na, nf = ablate_fn(m, x, y, extra)
            else:
                n, na, nf = ablate_fn(m, x, y)
            ns.append(n); nas.append(na); nfs.append(nf)
        if not ns:
            print(f"  [{label}] NO CHECKPOINTS FOUND  (tried {tmpl.format(seed='*')})")
            out[label] = None
        else:
            ns, nas, nfs = map(np.array, (ns, nas, nfs))
            out[label] = {
                "n": len(ns),
                "normal_mean": ns.mean(),     "normal_std": ns.std(),
                "no_attn_mean": nas.mean(),   "no_attn_std": nas.std(),
                "no_ffn_mean": nfs.mean(),    "no_ffn_std": nfs.std(),
            }
            print(f"  [{label}] n={len(ns):2d}  "
                  f"normal={ns.mean()*100:6.2f}%  "
                  f"no-attn={nas.mean()*100:6.2f}±{nas.std()*100:4.2f}%  "
                  f"no-ffn={nfs.mean()*100:6.2f}±{nfs.std()*100:4.2f}%")
    return out


def _plot(results, title, save_path, ylim=(0, 1.15)):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    width = 0.27
    xs = np.arange(len(VARIANT_ORDER))

    norm_means = []; norm_stds = []
    na_means   = []; na_stds   = []
    nf_means   = []; nf_stds   = []

    for v in VARIANT_ORDER:
        r = results.get(v)
        if r is None:
            for arr in (norm_means, norm_stds, na_means, na_stds, nf_means, nf_stds):
                arr.append(0.0)
        else:
            norm_means.append(r["normal_mean"]);  norm_stds.append(r["normal_std"])
            na_means.append(r["no_attn_mean"]);   na_stds.append(r["no_attn_std"])
            nf_means.append(r["no_ffn_mean"]);    nf_stds.append(r["no_ffn_std"])

    bars_norm = ax.bar(xs - width, norm_means, width, yerr=norm_stds, capsize=4,
           color=COLOR_NORMAL, edgecolor="black", linewidth=0.7,
           error_kw=dict(elinewidth=1.2, capthick=1.2), label="Normal")
    bars_na = ax.bar(xs,         na_means,   width, yerr=na_stds,   capsize=4,
           color=COLOR_NO_ATTN, edgecolor="black", linewidth=0.7,
           error_kw=dict(elinewidth=1.2, capthick=1.2), label="No Attention")
    bars_nf = ax.bar(xs + width, nf_means,   width, yerr=nf_stds,   capsize=4,
           color=COLOR_NO_FFN, edgecolor="black", linewidth=0.7,
           error_kw=dict(elinewidth=1.2, capthick=1.2), label="No FFN")

    def _annotate(bars, means, stds):
        for b, m, s in zip(bars, means, stds):
            top = m + s + 0.018
            ax.text(b.get_x() + b.get_width() / 2, top, f"{m*100:.0f}%",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color="#1A1A1A")
    # Normal bars are ~100% by construction; skip those labels to avoid
    # collision with the legend and reduce visual clutter.
    _annotate(bars_na, na_means, na_stds)
    _annotate(bars_nf, nf_means, nf_stds)

    ax.set_xticks(xs)
    ax.set_xticklabels(VARIANT_ORDER, fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylabel("Accuracy", fontsize=16, fontweight="bold")
    ax.set_ylim(ylim)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 1.0),
                     ncol=len(handles), frameon=True, fontsize=14,
                     handlelength=2.0, borderpad=0.6, columnspacing=2.5)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {save_path.relative_to(ROOT)}")


def run_add7():
    print("\n=== add-7 (matched widths, SiLU) ===")
    x, y, out_start = build_add7_data(num_digits=3)

    # Paper convention add-7: SiLU. Main runs use class default (SiLU).
    # GLU/MoE-GLU use h=170 (matched).
    templates = [
        ("FFN",     "add7_ffn_nonorm_s{seed}"),
        ("GLU",     "add7_glu_d170_silu_nonorm_s{seed}"),
        ("MoE",     "add7_moe_nonorm_s{seed}"),
        ("MoE-GLU", "add7_moe_glu_d170_silu_nonorm_s{seed}"),
    ]
    res = _gather(templates, ["best_model.pt"], ablate_add7, x, y, out_start)
    _plot(res, "Component Ablation on Add-7 (5 seeds, parameter-matched)",
          ROOT / "paper" / "images" / "add-7" / "MoE_redistributes_computation" / "ablation_+7_5_seeds.png")
    return res


def run_modadd():
    print("\n=== modadd (matched widths, GELU) ===")
    x, y = build_modadd_data(p=113)

    templates = [
        ("FFN",     "modadd_ffn_s{seed}"),
        ("GLU",     "modadd_glu_d340_s{seed}"),
        ("MoE",     "modadd_moe_s{seed}"),
        ("MoE-GLU", "modadd_moe_glu_d340_s{seed}"),
    ]
    res = _gather(templates, ["modadd_test99.pt", "modadd_best.pt"], ablate_modadd, x, y)
    _plot(res, "Component Ablation on (a+b) mod 113 (5 seeds, parameter-matched)",
          ROOT / "paper" / "images" / "mod-add" / "MoE_redistributes_computation" / "ablation_modular_addition_5_seeds.png")
    return res


def run_hist():
    print("\n=== histogram (matched widths, GELU) ===")
    x, y = build_hist_data(seed=0)

    templates = [
        ("FFN",     "hist_ffn_s{seed}"),
        ("GLU",     "hist_glu_d340_s{seed}"),
        ("MoE",     "hist_moe_s{seed}"),
        ("MoE-GLU", "hist_moe_glu_d340_s{seed}"),
    ]
    res = _gather(templates, ["hist_best.pt"], ablate_hist, x, y)
    _plot(res, "Component Ablation on Histogram (5 seeds, parameter-matched)",
          ROOT / "paper" / "images" / "hist" / "ablation_histogram_5_seeds.png")
    return res


def main():
    add7 = run_add7()
    modadd = run_modadd()
    hist = run_hist()

    print("\n\n" + "=" * 80)
    print("SUMMARY (matched widths)")
    print("=" * 80)
    for task, results in [("add-7", add7), ("modadd", modadd), ("hist", hist)]:
        print(f"\n[{task}]")
        for v in VARIANT_ORDER:
            r = results.get(v)
            if r is None:
                print(f"  {v:8s}: ---")
            else:
                print(f"  {v:8s}: n={r['n']}  normal={r['normal_mean']*100:6.2f}%  "
                      f"no-attn={r['no_attn_mean']*100:6.2f}±{r['no_attn_std']*100:4.2f}%  "
                      f"no-ffn={r['no_ffn_mean']*100:6.2f}±{r['no_ffn_std']*100:4.2f}%")


if __name__ == "__main__":
    main()
