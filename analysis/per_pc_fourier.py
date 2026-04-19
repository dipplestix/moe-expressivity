"""
Per-principal-component Fourier concentration analysis for all 4 FFN variants on modular addition.
Addresses reviewer concern: is the high top-PC concentration in one PC or spread across many?
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "model")
sys.path.insert(0, ".")

import numpy as np
import torch
from model import OneLayerTransformer
from data.modular_addition import ModularAdditionDataset

SEEDS = [42, 137, 256, 512, 1024]
P = 113
TOP_K = 10

model_keys = {'model_dim', 'num_heads', 'ffn_type', 'vocab_size', 'max_seq_len',
              'use_norm', 'is_causal', 'tie_embeddings', 'activation', 'dropout',
              'intermediate_dim', 'num_experts', 'top_k'}


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    cfg = {k: v for k, v in cfg.items() if k in model_keys}
    model = OneLayerTransformer(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def fourier_concentration(values, p):
    """Fraction of spectral power at the dominant frequency."""
    fft = np.abs(np.fft.fft(values))
    power = fft ** 2
    total = power.sum()
    if total == 0:
        return 0.0
    return float(power.max() / total)


def analyze_variant(ftype):
    all_concs = []      # shape: (n_seeds, TOP_K)
    all_variances = []  # shape: (n_seeds, TOP_K)

    for seed in SEEDS:
        path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_best.pt"
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            continue

        model, ckpt = load_model(path)

        # Use ALL p^2 pairs for cleaner Fourier analysis
        a = torch.arange(P).repeat_interleave(P)
        b = torch.arange(P).repeat(P)
        eq = torch.full_like(a, P)
        inputs = torch.stack([a, b, eq], dim=1)  # (P^2, 3)
        targets = ((a + b) % P).numpy()

        # Hook to capture FFN output
        hook_data = {}
        h = model.ffn.register_forward_hook(
            lambda m, inp, out: hook_data.update({"ffn_out": out.detach()}))
        with torch.no_grad():
            model(inputs)
        h.remove()

        acts = hook_data["ffn_out"][:, 2, :].numpy()  # (P^2, d_model) at '=' position

        # PCA
        acts_centered = acts - acts.mean(axis=0)
        U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)

        # Explained variance ratios
        var_explained = (S ** 2) / (S ** 2).sum()

        seed_concs = []
        seed_vars = []
        for k in range(min(TOP_K, Vt.shape[0])):
            proj = acts_centered @ Vt[k]  # (P^2,)

            # Group by (a+b) mod p: average projection for each target class
            by_target = np.zeros(P)
            counts = np.zeros(P)
            for i, t in enumerate(targets):
                by_target[t] += proj[i]
                counts[t] += 1
            mask = counts > 0
            by_target[mask] /= counts[mask]

            conc = fourier_concentration(by_target, P)
            seed_concs.append(conc)
            seed_vars.append(var_explained[k])

        all_concs.append(seed_concs)
        all_variances.append(seed_vars)

    return np.array(all_concs), np.array(all_variances)


def main():
    print("=" * 80)
    print("  PER-PRINCIPAL-COMPONENT FOURIER CONCENTRATION (modular addition, p=113)")
    print("  Averaged over 5 seeds: [42, 137, 256, 512, 1024]")
    print("=" * 80)

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        print(f"\n{'─' * 80}")
        print(f"  Variant: {ftype.upper()}")
        print(f"{'─' * 80}")

        concs, variances = analyze_variant(ftype)

        if len(concs) == 0:
            print("  No checkpoints found.")
            continue

        mean_concs = concs.mean(axis=0)
        std_concs = concs.std(axis=0)
        mean_vars = variances.mean(axis=0)
        std_vars = variances.std(axis=0)

        print(f"  Seeds found: {len(concs)}")
        print(f"  {'PC':<6} {'Expl. Var %':<20} {'Fourier Conc.':<20}")
        print(f"  {'─'*6} {'─'*20} {'─'*20}")
        cumul_var = 0.0
        for k in range(len(mean_concs)):
            cumul_var += mean_vars[k] * 100
            print(f"  PC{k:<3} {mean_vars[k]*100:>6.2f} +/- {std_vars[k]*100:>5.2f}  (cum {cumul_var:>5.1f}%)    {mean_concs[k]:.4f} +/- {std_concs[k]:.4f}")

        # Summary stats
        print(f"\n  Summary:")
        print(f"    Max single-PC Fourier conc:  {mean_concs.max():.4f}")
        print(f"    Mean top-10 Fourier conc:    {mean_concs.mean():.4f}")
        print(f"    PCs with conc > 0.10:        {(mean_concs > 0.10).sum()}")
        print(f"    PCs with conc > 0.20:        {(mean_concs > 0.20).sum()}")
        print(f"    Top-10 cumulative variance:  {mean_vars.sum()*100:.1f}%")


if __name__ == "__main__":
    main()
