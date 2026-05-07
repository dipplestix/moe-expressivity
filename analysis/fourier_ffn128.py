"""
Per-neuron Fourier concentration for dense FFN with intermediate_dim=128.

Control experiment: Does the GLU Fourier concentration drop (0.44 -> 0.07)
come from the multiplicative gate or from having fewer neurons (128 vs 256)?

Computes the same metric as analyze_all_variants.py:neuron_freq_analysis
for the FFN-128dim checkpoints across 5 seeds.
"""

import sys
import os
os.chdir("<PATH_TO_REPO>")
sys.path.insert(0, "model")

import numpy as np
import torch
from model import OneLayerTransformer
from data.modular_addition import ModularAdditionDataset

P = 113
DEVICE = "cpu"
SEEDS = [42, 137, 256, 512, 1024]


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    config = ckpt["config"]
    model = OneLayerTransformer(**config).to(DEVICE)
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


def neuron_fourier_concentration(activations, a_vals, b_vals, p):
    """
    Per-neuron Fourier concentration: for each neuron, group activations by
    (a+b) mod p, compute mean per group, FFT, fraction of power at dominant freq.
    Returns array of concentrations (one per neuron).
    """
    num_neurons = activations.shape[1]
    ab_sum = (a_vals + b_vals) % p

    # Average activation as function of (a+b) mod p
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

    return np.array(concentrations)


def main():
    print("=" * 60)
    print("  FFN-128dim Per-Neuron Fourier Concentration")
    print("  (Control for GLU neuron count confound)")
    print("=" * 60)

    inputs, targets, a_vals, b_vals = get_all_inputs()
    a_np = a_vals.numpy()
    b_np = b_vals.numpy()

    all_mean_concs = []
    all_median_concs = []

    for seed in SEEDS:
        ckpt_path = f"checkpoints/modadd_ffn_128dim_s{seed}/modadd_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"\n  Seed {seed}: checkpoint not found, skipping")
            continue

        model, ckpt = load_model(ckpt_path)
        test_acc = ckpt["test_acc"]
        epoch = ckpt["epoch"]

        # Hook to capture post-activation (after GELU) of up_proj
        hook_data = {}

        def hook_fn(module, input, output):
            hook_data["pre_act"] = output.detach()

        h = model.ffn.up_proj.register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = model(inputs)
        h.remove()

        # Check accuracy on full p^2
        preds = logits[:, 2, :].argmax(dim=-1)
        full_acc = (preds == targets).float().mean().item()

        # Get post-activation values at position 2 (the = token)
        pre_act = hook_data["pre_act"]
        # Shape depends on whether it's (batch, seq, dim) or (batch*seq, dim)
        if pre_act.dim() == 3:
            acts = pre_act[:, 2, :].numpy()
        else:
            # Reshape: (batch*3, dim) -> (batch, 3, dim) -> take pos 2
            acts = pre_act.view(-1, 3, pre_act.shape[-1])[:, 2, :].numpy()

        # Apply GELU to get post-activation (hook captures pre-activation output of linear)
        acts_post = torch.nn.functional.gelu(torch.tensor(acts)).numpy()

        concs = neuron_fourier_concentration(acts_post, a_np, b_np, P)

        mean_c = concs.mean()
        median_c = np.median(concs)
        all_mean_concs.append(mean_c)
        all_median_concs.append(median_c)

        print(f"\n  Seed {seed}: test_acc={test_acc:.4f} (epoch {epoch}), full_acc={full_acc:.4f}")
        print(f"    Neurons: {len(concs)}")
        print(f"    Mean concentration:   {mean_c:.4f}")
        print(f"    Median concentration: {median_c:.4f}")
        print(f"    Max concentration:    {concs.max():.4f}")
        print(f"    Min concentration:    {concs.min():.4f}")

    if all_mean_concs:
        print(f"\n{'='*60}")
        print(f"  SUMMARY (across {len(all_mean_concs)} seeds)")
        print(f"{'='*60}")
        print(f"  FFN-128dim mean concentration:  {np.mean(all_mean_concs):.4f} +/- {np.std(all_mean_concs):.4f}")
        print(f"  FFN-128dim median concentration: {np.mean(all_median_concs):.4f} +/- {np.std(all_median_concs):.4f}")
        print(f"\n  Comparison:")
        print(f"    Dense FFN (256 dim): ~0.44")
        print(f"    GLU       (128 dim): ~0.07")
        print(f"    Dense FFN (128 dim): {np.mean(all_mean_concs):.4f}  <-- this experiment")
        print(f"\n  Interpretation:")
        if np.mean(all_mean_concs) > 0.30:
            print(f"    FFN-128 shows HIGH concentration (~FFN-256).")
            print(f"    => GLU drop is due to GATING, not neuron count.")
        elif np.mean(all_mean_concs) < 0.15:
            print(f"    FFN-128 shows LOW concentration (~GLU).")
            print(f"    => GLU drop is a neuron count CONFOUND.")
        else:
            print(f"    FFN-128 shows INTERMEDIATE concentration.")
            print(f"    => Both gating and neuron count contribute.")


if __name__ == "__main__":
    main()
