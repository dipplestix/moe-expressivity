"""
Multi-seed version of the v2 diagnostic.

Runs probes across 5 seeds for each of {FFN, GLU, MoE, MoE-GLU}, aggregates
mean ± std, and prints a summary table. Goal: confirm whether the
FFN-leans-relation / GLU-leans-inventory split observed at seed 42 holds.
"""

import sys, os
os.chdir("<PATH_TO_REPO>")
sys.path.insert(0, ".")
sys.path.insert(0, "model")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import OneLayerTransformer
from data.histogram import HistogramDataset

DEVICE = "cpu"
MODEL_KEYS = {
    "model_dim", "num_heads", "ffn_type", "dropout", "vocab_size",
    "max_seq_len", "use_norm", "is_causal", "tie_embeddings",
    "activation", "intermediate_dim", "num_experts", "top_k",
}

SEEDS = [42, 137, 256, 512, 1024]
VARIANTS = [("ffn", "FFN"), ("glu", "GLU"), ("moe", "MoE"), ("moe_glu", "MoE-GLU")]


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    config = {k: v for k, v in ckpt["config"].items() if k in MODEL_KEYS}
    num_classes = ckpt["config"]["num_classes"]
    model = OneLayerTransformer(**config).to(DEVICE)
    model.unembed = nn.Linear(config["model_dim"], num_classes, bias=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, num_classes


def collect_residuals(model, inputs):
    B, L = inputs.shape
    pos = torch.arange(L, device=inputs.device)
    with torch.no_grad():
        x_embed = model.vocab(inputs) + model.pos_embed(pos)
        attn_out = model.atn(model.atn_norm(x_embed))
        x_post_attn = x_embed + attn_out
        ffn_out = model.ffn(model.ffn_norm(x_post_attn))
        x_post_ffn = x_post_attn + ffn_out
    return x_embed, x_post_attn, x_post_ffn


def linear_probe(features, targets, n_classes, train_frac=0.7, lr=0.1, steps=2000):
    N, D = features.shape
    n_train = int(N * train_frac)
    Xtr, ytr = features[:n_train], targets[:n_train]
    Xte, yte = features[n_train:], targets[n_train:]

    probe = nn.Linear(D, n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        logits = probe(Xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_acc = (probe(Xte).argmax(-1) == yte).float().mean().item()
    return test_acc


def probe1_residual_readout(model, inputs, targets, n_classes):
    x_embed, x_post_attn, x_post_ffn = collect_residuals(model, inputs)
    B, L, D = x_embed.shape
    feats_e = x_embed.reshape(B * L, D)
    feats_a = x_post_attn.reshape(B * L, D)
    feats_f = x_post_ffn.reshape(B * L, D)
    targs = targets.reshape(B * L)
    e = linear_probe(feats_e, targs, n_classes)
    a = linear_probe(feats_a, targs, n_classes)
    f = linear_probe(feats_f, targs, n_classes)
    return {"embed": e, "post_attn": a, "post_ffn": f}


def probe2_wup_token_selectivity(model, T_vocab):
    ffn = model.ffn
    E = model.vocab.weight[:T_vocab]
    with torch.no_grad():
        E_norm = model.ffn_norm(E)

    if hasattr(ffn, "experts"):
        per_expert = []
        for e in ffn.experts:
            with torch.no_grad():
                if hasattr(e, "gate_proj"):
                    h = e.activation(e.gate_proj(E_norm)) * e.up_proj(E_norm)
                else:
                    h = e.activation(e.up_proj(E_norm))
            per_expert.append(h)
        H = torch.cat(per_expert, dim=-1)
    elif hasattr(ffn, "gate_proj"):
        with torch.no_grad():
            H = ffn.activation(ffn.gate_proj(E_norm)) * ffn.up_proj(E_norm)
    else:
        with torch.no_grad():
            H = ffn.activation(ffn.up_proj(E_norm))

    H_np = H.numpy()
    abs_H = np.abs(H_np)
    selectivity = abs_H.max(axis=0) / (abs_H.mean(axis=0) + 1e-8)
    preferred = abs_H.argmax(axis=0)
    n_neurons = H_np.shape[1]
    counts = np.bincount(preferred, minlength=T_vocab)
    return {
        "mean_selectivity": float(selectivity.mean()),
        "frac_sel_gt5": float((selectivity > 5).sum() / n_neurons),
        "frac_sel_gt2": float((selectivity > 2).sum() / n_neurons),
        "token_coverage": int((counts > 0).sum()),
    }


def main():
    dataset = HistogramDataset(T=32, L=10, n_train=1000, n_test=2000, seed=42, device=DEVICE)
    inputs = dataset.test_inputs
    targets = dataset.test_targets

    print("=" * 96)
    print("Histogram counting strategy diagnostic v3 (5 seeds per variant)")
    print("=" * 96)

    all_results = {}
    for ftype, label in VARIANTS:
        print(f"\n--- {label} ({ftype}) ---")
        per_seed = []
        for seed in SEEDS:
            path = f"checkpoints/hist_{ftype}_s{seed}/hist_best.pt"
            if not os.path.exists(path):
                print(f"  s{seed}: MISSING")
                continue
            model, ckpt, n_classes = load_model(path)
            p1 = probe1_residual_readout(model, inputs, targets, n_classes)
            p2 = probe2_wup_token_selectivity(model, T_vocab=32)
            row = {
                "seed": seed,
                "test_acc": ckpt["test_acc"],
                **{f"p1_{k}": v for k, v in p1.items()},
                **{f"p2_{k}": v for k, v in p2.items()},
            }
            per_seed.append(row)
            print(f"  s{seed}: post_attn={p1['post_attn']:.3f} post_ffn={p1['post_ffn']:.3f} "
                  f"sel_mean={p2['mean_selectivity']:.2f} frac_sel>5={p2['frac_sel_gt5']:.3f}")
        all_results[ftype] = per_seed

    # Aggregate
    print("\n" + "=" * 96)
    print("AGGREGATED (mean ± std across 5 seeds)")
    print("=" * 96)
    header = (f"{'variant':10s}  {'embed_acc':>11s}  {'post_attn':>13s}  "
              f"{'post_ffn':>13s}  {'attn_lift':>11s}  {'ffn_lift':>11s}  "
              f"{'sel_mean':>10s}  {'frac>5':>8s}")
    print(header)
    print("-" * len(header))
    summary = {}
    for ftype, label in VARIANTS:
        rows = all_results.get(ftype, [])
        if not rows:
            continue
        e = np.array([r["p1_embed"] for r in rows])
        a = np.array([r["p1_post_attn"] for r in rows])
        f = np.array([r["p1_post_ffn"] for r in rows])
        sel = np.array([r["p2_mean_selectivity"] for r in rows])
        fr5 = np.array([r["p2_frac_sel_gt5"] for r in rows])
        attn_lift = a - e
        ffn_lift = f - a
        summary[ftype] = {
            "embed":     (e.mean(), e.std()),
            "post_attn": (a.mean(), a.std()),
            "post_ffn":  (f.mean(), f.std()),
            "attn_lift": (attn_lift.mean(), attn_lift.std()),
            "ffn_lift":  (ffn_lift.mean(), ffn_lift.std()),
            "sel_mean":  (sel.mean(), sel.std()),
            "frac_gt5":  (fr5.mean(), fr5.std()),
            "n":         len(rows),
        }
        s = summary[ftype]
        print(f"{label:10s}  "
              f"{s['embed'][0]:.3f}±{s['embed'][1]:.3f}  "
              f"{s['post_attn'][0]:.3f}±{s['post_attn'][1]:.3f}  "
              f"{s['post_ffn'][0]:.3f}±{s['post_ffn'][1]:.3f}  "
              f"{s['attn_lift'][0]:+.3f}±{s['attn_lift'][1]:.3f}  "
              f"{s['ffn_lift'][0]:+.3f}±{s['ffn_lift'][1]:.3f}  "
              f"{s['sel_mean'][0]:.2f}±{s['sel_mean'][1]:.2f}  "
              f"{s['frac_gt5'][0]:.2f}±{s['frac_gt5'][1]:.2f}")

    # Strategy diagnosis (per-seed, then majority)
    print("\nPer-seed strategy classification:")
    print(f"  RELATION = attn_lift > ffn_lift  (count appears in attention output)")
    print(f"  INVENTORY = ffn_lift > attn_lift (count built by FFN)")
    for ftype, label in VARIANTS:
        rows = all_results.get(ftype, [])
        if not rows: continue
        labels = []
        for r in rows:
            al = r["p1_post_attn"] - r["p1_embed"]
            fl = r["p1_post_ffn"] - r["p1_post_attn"]
            labels.append("REL" if al > fl else "INV")
        print(f"  {label:10s} (n={len(rows)}): {' '.join(labels)}")

    # Save
    np.savez("figures/histogram_strategy_v3.npz",
             results={k: v for k, v in all_results.items()},
             summary=summary)
    print("\nSaved figures/histogram_strategy_v3.npz")


if __name__ == "__main__":
    main()
