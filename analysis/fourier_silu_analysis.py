"""
Fourier concentration analysis: SiLU vs GELU for GLU and MoE-GLU on modular addition.
Computes per-neuron concentration and top-PC concentration.
"""

import sys
import os
os.chdir("<PATH_TO_REPO>")
sys.path.insert(0, ".")
sys.path.insert(0, "model")

import numpy as np
import torch
from model import OneLayerTransformer
from data.modular_addition import ModularAdditionDataset

P = 113
DEVICE = "cpu"
SEEDS = [42, 137, 256, 512, 1024]


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = ckpt["config"]
    model_keys = {'model_dim', 'num_heads', 'ffn_type', 'vocab_size', 'max_seq_len',
                  'use_norm', 'is_causal', 'tie_embeddings', 'activation', 'dropout',
                  'intermediate_dim', 'num_experts', 'top_k'}
    cfg = {k: v for k, v in cfg.items() if k in model_keys}
    model = OneLayerTransformer(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def get_all_inputs(p=P):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    inputs = torch.stack([a, b, eq], dim=1)
    targets = (a + b) % p
    return inputs, targets, a, b


def get_neuron_activations(model, inputs, ffn_type):
    """Get post-activation hidden neuron activations."""
    ffn_in_data = {}
    h = model.ffn.register_forward_pre_hook(lambda m, inp: ffn_in_data.update({"x": inp[0].detach()}))
    with torch.no_grad():
        model(inputs)
    h.remove()
    ffn_in = ffn_in_data["x"][:, 2, :]  # at = position

    if ffn_type in ("ffn",):
        with torch.no_grad():
            act = model.ffn.activation(model.ffn.up_proj(ffn_in)).numpy()
        return act
    elif ffn_type in ("glu",):
        with torch.no_grad():
            gate = model.ffn.activation(model.ffn.gate_proj(ffn_in))
            up = model.ffn.up_proj(ffn_in)
            act = (gate * up).numpy()
        return act
    elif ffn_type in ("moe", "moe_glu"):
        # Concatenate all expert activations
        all_acts = []
        for expert in model.ffn.experts:
            with torch.no_grad():
                if hasattr(expert, "gate_proj"):
                    gate = expert.activation(expert.gate_proj(ffn_in))
                    up = expert.up_proj(ffn_in)
                    act = (gate * up).numpy()
                else:
                    act = expert.activation(expert.up_proj(ffn_in)).numpy()
            all_acts.append(act)
        # Return concatenated (N, total_hidden)
        return np.concatenate(all_acts, axis=1)


def neuron_fourier_concentration(activations, a_np, b_np, p):
    """Per-neuron Fourier concentration: for each neuron, average activation by (a+b)%p,
    FFT, fraction of power at dominant frequency."""
    num_neurons = activations.shape[1]
    ab_sum = (a_np + b_np) % p
    neuron_by_sum = np.zeros((p, num_neurons))
    for s in range(p):
        mask = ab_sum == s
        if mask.sum() > 0:
            neuron_by_sum[s] = activations[mask].mean(axis=0)

    concentrations = []
    for n in range(num_neurons):
        spectrum = np.abs(np.fft.fft(neuron_by_sum[:, n])) ** 2
        spectrum[0] = 0  # remove DC
        total = spectrum.sum()
        if total < 1e-12:
            concentrations.append(0.0)
        else:
            concentrations.append(float(np.max(spectrum) / total))
    return np.mean(concentrations)


def top_pc_fourier_concentration(activations, targets_np, p, top_k=5):
    """PCA on activations, compute Fourier concentration of top PCs."""
    acts_centered = activations - activations.mean(axis=0)
    U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)

    pc_concs = []
    for k in range(min(top_k, Vt.shape[0])):
        proj = acts_centered @ Vt[k]
        # Group by (a+b) mod p
        by_target = np.zeros(p)
        counts = np.zeros(p)
        for i, t in enumerate(targets_np):
            by_target[t] += proj[i]
            counts[t] += 1
        mask = counts > 0
        by_target[mask] /= counts[mask]

        fft_power = np.abs(np.fft.fft(by_target)) ** 2
        fft_power[0] = 0  # remove DC
        total = fft_power.sum()
        conc = fft_power.max() / total if total > 0 else 0
        pc_concs.append(conc)
    return pc_concs


def analyze_config(ftype, activation_name, skip_seeds=None):
    """Analyze all seeds for a given (ftype, activation) config."""
    skip_seeds = skip_seeds or set()

    if activation_name == "gelu":
        prefix = f"modadd_{ftype}"
    else:
        prefix = f"modadd_{ftype}_{activation_name}"

    neuron_concs = []
    pc1_concs = []
    top5_pc_concs = []
    valid_seeds = []

    for seed in SEEDS:
        if seed in skip_seeds:
            continue
        path = f"checkpoints/{prefix}_s{seed}/modadd_best.pt"
        if not os.path.exists(path):
            continue

        model, ckpt = load_model(path)
        test_acc = ckpt.get("test_acc", 0)
        if test_acc < 0.5:
            print(f"  Skipping {path}: test_acc={test_acc:.3f}")
            continue

        inputs, targets, a, b = get_all_inputs(P)
        a_np, b_np = a.numpy(), b.numpy()
        targets_np = targets.numpy()

        act = get_neuron_activations(model, inputs, ftype)

        nc = neuron_fourier_concentration(act, a_np, b_np, P)
        pc_concs = top_pc_fourier_concentration(act, targets_np, P, top_k=5)

        neuron_concs.append(nc)
        pc1_concs.append(pc_concs[0])
        top5_pc_concs.append(max(pc_concs))
        valid_seeds.append(seed)

    return {
        "neuron_conc": neuron_concs,
        "pc1_conc": pc1_concs,
        "top5_pc_conc": top5_pc_concs,
        "n_seeds": len(valid_seeds),
        "seeds": valid_seeds,
    }


def fmt(vals):
    if not vals:
        return "N/A"
    return f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}"


if __name__ == "__main__":
    configs = [
        ("glu", "gelu", set()),
        ("glu", "silu", {1024}),  # skip failed seed
        ("moe_glu", "gelu", set()),
        ("moe_glu", "silu", set()),
    ]

    # Also include FFN baselines for reference
    ffn_configs = [
        ("ffn", "gelu", set()),
        ("moe", "gelu", set()),
    ]

    all_results = {}
    print("=" * 70)
    print("  FOURIER CONCENTRATION: SiLU vs GELU COMPARISON")
    print("=" * 70)

    for ftype, act_name, skip in ffn_configs + configs:
        label = f"{ftype} ({act_name})"
        print(f"\nAnalyzing {label}...")
        res = analyze_config(ftype, act_name, skip)
        all_results[label] = res

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Config':<22} {'N':>3}  {'Per-Neuron Conc':>20}  {'Top-PC1 Conc':>20}  {'Max top-5 PC':>20}")
    print("-" * 70)
    for label, res in all_results.items():
        print(f"{label:<22} {res['n_seeds']:>3}  {fmt(res['neuron_conc']):>20}  {fmt(res['pc1_conc']):>20}  {fmt(res['top5_pc_conc']):>20}")

    # Print the key comparison
    print("\n" + "=" * 70)
    print("  KEY COMPARISON: Does H2 (GLU opacity) hold under SiLU?")
    print("=" * 70)
    for ftype in ["glu", "moe_glu"]:
        gelu = all_results.get(f"{ftype} (gelu)")
        silu = all_results.get(f"{ftype} (silu)")
        if gelu and silu and gelu["neuron_conc"] and silu["neuron_conc"]:
            print(f"\n  {ftype.upper()}:")
            print(f"    GELU per-neuron: {fmt(gelu['neuron_conc'])},  top-PC: {fmt(gelu['top5_pc_conc'])}")
            print(f"    SiLU per-neuron: {fmt(silu['neuron_conc'])},  top-PC: {fmt(silu['top5_pc_conc'])}")
            gelu_ratio = np.mean(gelu['top5_pc_conc']) / max(np.mean(gelu['neuron_conc']), 1e-6)
            silu_ratio = np.mean(silu['top5_pc_conc']) / max(np.mean(silu['neuron_conc']), 1e-6)
            print(f"    GELU PC/neuron ratio: {gelu_ratio:.1f}x")
            print(f"    SiLU PC/neuron ratio: {silu_ratio:.1f}x")
    print()
