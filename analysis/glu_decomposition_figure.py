"""
Enhanced GLU decomposition figure showing how multiplicative gating
destroys per-neuron Fourier structure.

Three panels:
(a) Example neuron Fourier spectra: gate vs up vs product for a single neuron
(b) Fourier concentration distribution: histogram for gate, up, product
(c) Summary bar chart: mean concentration with FFN reference line
"""

import sys
import os
os.chdir("<PATH_TO_REPO>")
sys.path.insert(0, ".")
sys.path.insert(0, "model")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

from model import OneLayerTransformer

P = 113
DEVICE = "cpu"


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = OneLayerTransformer(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def get_neuron_fourier(activations, a_np, b_np, p):
    """Get per-neuron Fourier spectrum as function of (a+b) mod p."""
    num_neurons = activations.shape[1]
    ab_sum = (a_np + b_np) % p

    neuron_by_sum = np.zeros((p, num_neurons))
    for s in range(p):
        mask = ab_sum == s
        if mask.sum() > 0:
            neuron_by_sum[s] = activations[mask].mean(axis=0)

    spectra = np.zeros((num_neurons, p))
    concentrations = np.zeros(num_neurons)
    dominant_freqs = np.zeros(num_neurons, dtype=int)

    for n in range(num_neurons):
        spectrum = np.abs(np.fft.fft(neuron_by_sum[:, n])) ** 2
        spectrum[0] = 0
        spectra[n] = spectrum
        total = spectrum.sum()
        if total > 1e-12:
            concentrations[n] = float(np.max(spectrum) / total)
            dominant_freqs[n] = int(np.argmax(spectrum))

    return spectra, concentrations, dominant_freqs


def make_figure():
    # Setup inputs
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    eq = torch.full_like(a, P)
    inputs = torch.stack([a, b, eq], dim=1)
    a_np, b_np = a.numpy(), b.numpy()

    # Load GLU model (seed 42)
    model, ckpt = load_model("checkpoints/modadd_glu_s42/modadd_best.pt")

    # Get FFN input
    ffn_in_data = {}
    h = model.ffn.register_forward_pre_hook(
        lambda m, inp: ffn_in_data.update({"x": inp[0].detach()}))
    with torch.no_grad():
        model(inputs)
    h.remove()
    ffn_in = ffn_in_data["x"][:, 2, :]  # = position

    # Compute components
    with torch.no_grad():
        gate = model.ffn.activation(model.ffn.gate_proj(ffn_in)).numpy()
        up = model.ffn.up_proj(ffn_in).numpy()
        product = (model.ffn.activation(model.ffn.gate_proj(ffn_in)) * model.ffn.up_proj(ffn_in)).numpy()

    # Also load FFN model for reference
    model_ffn, _ = load_model("checkpoints/modadd_ffn_s42/modadd_best.pt")
    ffn_in_data2 = {}
    h2 = model_ffn.ffn.register_forward_pre_hook(
        lambda m, inp: ffn_in_data2.update({"x": inp[0].detach()}))
    with torch.no_grad():
        model_ffn(inputs)
    h2.remove()
    ffn_in2 = ffn_in_data2["x"][:, 2, :]
    with torch.no_grad():
        ffn_act = model_ffn.ffn.activation(model_ffn.ffn.up_proj(ffn_in2)).numpy()

    # Compute Fourier stats
    gate_spectra, gate_concs, gate_freqs = get_neuron_fourier(gate, a_np, b_np, P)
    up_spectra, up_concs, up_freqs = get_neuron_fourier(up, a_np, b_np, P)
    prod_spectra, prod_concs, prod_freqs = get_neuron_fourier(product, a_np, b_np, P)
    ffn_spectra, ffn_concs, ffn_freqs = get_neuron_fourier(ffn_act, a_np, b_np, P)

    # Find a neuron where gate and up have structure but product doesn't
    # Best: neuron where gate_conc and up_conc are both > 0.1 but prod_conc < 0.05
    candidates = []
    for n in range(len(gate_concs)):
        if gate_concs[n] > 0.1 and up_concs[n] > 0.1 and prod_concs[n] < 0.1:
            candidates.append((n, gate_concs[n], up_concs[n], prod_concs[n]))
    candidates.sort(key=lambda x: x[1] + x[2], reverse=True)

    if not candidates:
        # Fallback: just pick the neuron with highest gate-product gap
        gaps = gate_concs - prod_concs
        example_neuron = int(np.argmax(gaps))
    else:
        example_neuron = candidates[0][0]

    # ============================================================
    # Create figure
    # ============================================================
    fig = plt.figure(figsize=(18, 5))

    # Panel (a): Example neuron Fourier spectra
    ax1 = fig.add_subplot(131)
    freqs = np.arange(P // 2 + 1)

    def fold_spectrum(spectrum):
        folded = spectrum[:P // 2 + 1].copy()
        folded[1:] += spectrum[P // 2 + 1:][::-1][:len(folded) - 1]
        return folded / folded.sum() if folded.sum() > 0 else folded

    ax1.plot(freqs, fold_spectrum(gate_spectra[example_neuron]),
             color='#e74c3c', label=f'Gate (conc={gate_concs[example_neuron]:.2f})', linewidth=1.5, alpha=0.8)
    ax1.plot(freqs, fold_spectrum(up_spectra[example_neuron]),
             color='#3498db', label=f'Up (conc={up_concs[example_neuron]:.2f})', linewidth=1.5, alpha=0.8)
    ax1.plot(freqs, fold_spectrum(prod_spectra[example_neuron]),
             color='#2c3e50', label=f'Product (conc={prod_concs[example_neuron]:.2f})', linewidth=2)

    ax1.set_xlabel("Frequency (folded)")
    ax1.set_ylabel("Normalized Power")
    ax1.set_title(f"(a) Fourier Spectrum of Neuron {example_neuron}")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, P // 2)
    ax1.grid(True, alpha=0.2)

    # Panel (b): Concentration distributions
    ax2 = fig.add_subplot(132)
    bins = np.linspace(0, 1, 31)
    ax2.hist(gate_concs, bins=bins, alpha=0.5, color='#e74c3c', label='Gate', edgecolor='black', linewidth=0.3)
    ax2.hist(up_concs, bins=bins, alpha=0.5, color='#3498db', label='Up', edgecolor='black', linewidth=0.3)
    ax2.hist(prod_concs, bins=bins, alpha=0.7, color='#2c3e50', label='Product', edgecolor='black', linewidth=0.3)

    ax2.axvline(x=np.mean(ffn_concs), color='#1f77b4', linestyle='--', linewidth=2,
                label=f'FFN mean ({np.mean(ffn_concs):.2f})')
    ax2.set_xlabel("Fourier Concentration")
    ax2.set_ylabel("# Neurons")
    ax2.set_title("(b) Concentration Distribution (GLU)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # Panel (c): Summary bar chart
    ax3 = fig.add_subplot(133)
    components = ['FFN\nneurons', 'GLU\ngate', 'GLU\nup', 'GLU\ngate×up']
    means = [np.mean(ffn_concs), np.mean(gate_concs), np.mean(up_concs), np.mean(prod_concs)]
    colors = ['#1f77b4', '#e74c3c', '#3498db', '#2c3e50']

    bars = ax3.bar(range(4), means, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(components)
    ax3.set_ylabel("Mean Fourier Concentration")
    ax3.set_title("(c) Multiplicative Destruction")
    ax3.grid(True, alpha=0.2, axis='y')
    ax3.set_ylim(0, 0.55)

    # Bar value labels (data labels, not annotations).
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    # NOTE: superseded by analysis/make_matched_figures.py:fig_glu_decomposition_matched
    # which uses the parameter-matched checkpoints (h=340) and writes the same path.
    fig.suptitle("How GLU Destroys Per-Neuron Fourier Structure\n"
                 "(Modular Addition, seed 42)", fontsize=15, y=1.05)
    fig.tight_layout()
    fig.savefig("figures/fig_glu_decomposition_enhanced.png")
    print("Saved fig_glu_decomposition_enhanced.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    make_figure()
    print("Done!")
