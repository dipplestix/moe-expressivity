"""Re-evaluate the random-routing experiment checkpoints with the same
digit-only metric used by `make_matched_ablation_figures.py` (Set A's metric).

Goal: produce a unified Tab:random-routing where Dense FFN and MoE values
match the matched-ablation Fig 2 / Sec 4.1 numbers, eliminating the 1-2 pp
mismatch between Sec 4.1 (+45.9 pp) and Sec 4.2 (+45.4 pp).

Metric: digit-only no-FFN / no-attn accuracy on the add-7 output positions
(slice(out_start, -1), excluding the trailing PAD prediction position),
identical to `analysis/make_matched_ablation_figures.py:ablate_add7`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("<PATH_TO_REPO>")
sys.path.insert(0, str(ROOT))

from analysis.analyze_activation_symmetry import SEEDS, load_model, build_add7_data
from analysis.make_matched_ablation_figures import ablate_add7


# (label, dir_template, ckpt_filename) — every condition in Tab:random-routing.
CONDITIONS = [
    ("Dense FFN",                    "add7_ffn_nonorm_s{seed}",                "best_model.pt"),
    ("Narrow FFN (h=64, no routing)", "add7_ffn_narrow_nonorm_s{seed}",         "best_model.pt"),
    ("MoE (learned routing)",        "add7_moe_nonorm_s{seed}",                "best_model.pt"),
    ("MoE (random routing)",         "add7_moe_randroute_nonorm_s{seed}",      "best_model.pt"),
    ("MoE-GLU (learned routing)",    "add7_moe_glu_nonorm_s{seed}",            "best_model.pt"),
    ("MoE-GLU (random routing)",     "add7_moe_glu_randroute_nonorm_s{seed}",  "best_model.pt"),
]


def main():
    x, y, out_start = build_add7_data(num_digits=3)
    print(f"add-7 eval: digit-only metric, slice(out_start={out_start}, -1)\n")
    print(f"{'Variant':<40s}  {'n':>3s}  {'Normal':>10s}  {'No-attn':>14s}  {'No-FFN':>14s}")
    print("-" * 95)
    for label, tmpl, ckpt_name in CONDITIONS:
        ns, nas, nfs = [], [], []
        for s in SEEDS:
            ck = ROOT / "checkpoints" / tmpl.format(seed=s) / ckpt_name
            if not ck.exists():
                print(f"  [{label}] missing {ck.parent.name}")
                continue
            try:
                m, _, _ = load_model(ck)
            except Exception as e:
                print(f"  [{label}] load-err {ck.parent.name}: {e}")
                continue
            n, na, nf = ablate_add7(m, x, y, out_start)
            ns.append(n); nas.append(na); nfs.append(nf)
        if not ns:
            continue
        ns, nas, nfs = map(np.array, (ns, nas, nfs))
        print(f"{label:<40s}  {len(ns):>3d}  "
              f"{ns.mean()*100:6.2f}%        "
              f"{nas.mean()*100:5.2f} ± {nas.std()*100:4.2f}%   "
              f"{nfs.mean()*100:5.2f} ± {nfs.std()*100:4.2f}%")


if __name__ == "__main__":
    main()
