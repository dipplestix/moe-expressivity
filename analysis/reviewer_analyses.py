"""
Reviewer-requested analyses that run on existing checkpoints.
1. Chance MI baseline for expert routing
2. GLU subspace analysis (PCA on activations)
3. Histogram error modes under no-FFN ablation
"""

import sys
import os
os.chdir("<PATH_TO_REPO>")
sys.path.insert(0, "model")

import numpy as np
import torch
from model import OneLayerTransformer
from collections import Counter

SEEDS = [42, 137, 256, 512, 1024]
NUM_DIGITS = 3
EOS_TOKEN = 11


def num_to_reversed_digits(n, nd):
    d = []
    for _ in range(nd):
        d.append(n % 10)
        n //= 10
    return d


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    # Remove non-model keys that some checkpoints store
    model_keys = {'model_dim', 'num_heads', 'ffn_type', 'vocab_size', 'max_seq_len',
                  'use_norm', 'is_causal', 'tie_embeddings', 'activation', 'dropout',
                  'intermediate_dim', 'num_experts', 'top_k'}
    cfg = {k: v for k, v in cfg.items() if k in model_keys}
    model = OneLayerTransformer(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ============================================================
# 1. Chance MI baseline
# ============================================================
def chance_mi_baseline():
    """Compute expected MI under random routing given operation distribution."""
    print("=" * 60)
    print("  CHANCE MI BASELINE")
    print("=" * 60)

    # Operation distribution for 3-digit add-7
    ops = []
    for n in range(1000):
        carry = 0
        temp = n
        for d in range(NUM_DIGITS + 1):
            digit = temp % 10 if d < NUM_DIGITS else 0
            temp //= 10
            s = digit + (7 if d == 0 else carry)
            if d == 0:
                ops.append('+7')
            elif carry == 1:
                ops.append('+1')
            else:
                ops.append('+0')
            carry = 1 if s >= 10 else 0

    op_counts = Counter(ops)
    total = len(ops)
    print(f"Operation distribution: {dict(op_counts)}")
    print(f"Total tokens: {total}")

    # Under random uniform routing to E=4 experts,
    # expert assignment is independent of operation type.
    # MI(expert, operation) = 0 exactly.
    # Normalized MI is also 0.
    # But with finite samples, we get nonzero MI from noise.

    n_simulations = 1000
    E = 4
    mis = []
    for _ in range(n_simulations):
        # Random routing
        assignments = np.random.randint(0, E, size=total)
        # Compute MI
        joint = np.zeros((3, E))
        for op, exp in zip(ops, assignments):
            oi = {'+7': 0, '+1': 1, '+0': 2}[op]
            joint[oi, exp] += 1
        joint /= joint.sum()
        p_op = joint.sum(axis=1, keepdims=True)
        p_exp = joint.sum(axis=0, keepdims=True)

        mi = 0
        for i in range(3):
            for j in range(E):
                if joint[i, j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (p_op[i, 0] * p_exp[0, j]))

        # Normalize
        h_op = -np.sum(p_op * np.log2(p_op + 1e-10))
        h_exp = -np.sum(p_exp * np.log2(p_exp + 1e-10))
        nmi = mi / min(h_op, h_exp) if min(h_op, h_exp) > 0 else 0
        mis.append(nmi)

    print(f"\nChance MI (random routing, {n_simulations} simulations):")
    print(f"  Mean NMI: {np.mean(mis):.4f} +/- {np.std(mis):.4f}")
    print(f"  Max NMI:  {np.max(mis):.4f}")
    print(f"\nObserved MoE-GLU NMI: 0.28 +/- 0.20")
    print(f"Observed MoE NMI: 0.26 +/- 0.18")
    print(f"Chance-corrected: MoE-GLU = {0.28 - np.mean(mis):.3f}, MoE = {0.26 - np.mean(mis):.3f}")


# ============================================================
# 2. GLU subspace analysis (PCA)
# ============================================================
def glu_pca_analysis():
    """PCA on GLU vs FFN activations to check if Fourier structure
    exists at the subspace level even when per-neuron concentration is low."""
    from data.modular_addition import ModularAdditionDataset

    print("\n" + "=" * 60)
    print("  GLU SUBSPACE ANALYSIS (PCA)")
    print("=" * 60)

    P = 113
    device = "cpu"

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        concentrations = []
        for seed in SEEDS:
            path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_best.pt"
            if not os.path.exists(path):
                continue

            model, ckpt = load_model(path)
            dataset = ModularAdditionDataset(p=P, seed=seed, device=device)

            # Get FFN output activations
            hook_data = {}
            h = model.ffn.register_forward_hook(
                lambda m, inp, out: hook_data.update({"ffn_out": out.detach()}))
            with torch.no_grad():
                model(dataset.test_inputs)
            h.remove()

            acts = hook_data["ffn_out"][:, 2, :].numpy()  # (N, d_model) at = position

            # PCA
            acts_centered = acts - acts.mean(axis=0)
            U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)

            # Project onto top-k PCs and measure Fourier concentration of each
            targets = dataset.test_targets.numpy()
            top_k = 10
            pc_fourier_concs = []
            for k in range(min(top_k, Vt.shape[0])):
                proj = acts_centered @ Vt[k]  # projection onto PC k
                # Group by (a+b) mod p and compute Fourier concentration
                by_target = np.zeros(P)
                counts = np.zeros(P)
                for i, t in enumerate(targets):
                    by_target[t] += proj[i]
                    counts[t] += 1
                mask = counts > 0
                by_target[mask] /= counts[mask]

                fft = np.abs(np.fft.fft(by_target))
                fft_power = fft ** 2
                conc = fft_power.max() / fft_power.sum() if fft_power.sum() > 0 else 0
                pc_fourier_concs.append(conc)

            concentrations.append(pc_fourier_concs)

        if concentrations:
            mean_concs = np.mean(concentrations, axis=0)
            print(f"\n  {ftype}: Top-10 PC Fourier concentrations (mean across {len(concentrations)} seeds):")
            for k, c in enumerate(mean_concs):
                print(f"    PC{k}: {c:.3f}")
            print(f"    Max PC concentration: {np.max(mean_concs):.3f}")
            print(f"    Per-neuron concentration: {'0.44' if ftype=='ffn' else '0.47' if ftype=='moe' else '0.07' if ftype=='glu' else '0.18'}")


# ============================================================
# 3. Histogram error modes under no-FFN ablation
# ============================================================
def histogram_error_modes():
    """Analyze which counts fail under no-FFN ablation on histogram task."""
    from data.histogram import HistogramDataset

    print("\n" + "=" * 60)
    print("  HISTOGRAM ERROR MODES (no-FFN ablation)")
    print("=" * 60)

    device = "cpu"

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        count_accs_normal = {c: [] for c in range(10)}
        count_accs_noffn = {c: [] for c in range(10)}

        for seed in SEEDS:
            path = f"checkpoints/hist_{ftype}_s{seed}/hist_best.pt"
            if not os.path.exists(path):
                continue

            model, ckpt = load_model(path)
            dataset = HistogramDataset(T=32, L=10, n_test=3000, seed=seed, device=device)
            inputs = dataset.test_inputs
            targets = dataset.test_targets

            # Normal accuracy by count
            with torch.no_grad():
                logits = model(inputs)
            preds = logits.argmax(-1)

            for c in range(10):
                mask = (targets == c)
                if mask.any():
                    acc = (preds[mask] == targets[mask]).float().mean().item()
                    count_accs_normal[c].append(acc)

            # No-FFN accuracy by count
            orig = model.ffn.forward
            model.ffn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
            with torch.no_grad():
                logits_nf = model(inputs)
            preds_nf = logits_nf.argmax(-1)
            model.ffn.forward = orig

            for c in range(10):
                mask = (targets == c)
                if mask.any():
                    acc = (preds_nf[mask] == targets[mask]).float().mean().item()
                    count_accs_noffn[c].append(acc)

        print(f"\n  {ftype}:")
        print(f"  {'Count':<8} {'Normal':<15} {'No-FFN':<15} {'Drop':<10}")
        print(f"  {'-'*48}")
        for c in range(10):
            if count_accs_normal[c]:
                n_mean = np.mean(count_accs_normal[c])
                nf_mean = np.mean(count_accs_noffn[c])
                print(f"  {c+1:<8} {n_mean:.1%}           {nf_mean:.1%}           {n_mean - nf_mean:+.1%}")


if __name__ == "__main__":
    chance_mi_baseline()
    glu_pca_analysis()
    histogram_error_modes()
