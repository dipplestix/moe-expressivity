"""
Pearce et al. (2024) weight-based analysis of GLU models.

Express GLU as bilinear form: output_i = sum_j,k T_ijk * x_j * x_k
where T_ijk = sum_m W_down[i,m] * W_gate[m,j] * W_up[m,k]

Eigendecompose T to find interpretable structure.
On modular addition: check if eigenvectors correspond to Fourier frequencies.
On add-7: exploratory — what structure do the weights reveal?
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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
SEEDS = [42, 137, 256, 512, 1024]


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = OneLayerTransformer(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def get_bilinear_tensor(glu_module):
    """Construct the bilinear tensor T_ijk = sum_m W_down[i,m] * W_gate[m,j] * W_up[m,k]"""
    W_down = glu_module.down_proj.weight.detach()  # (out_dim, intermediate)
    W_gate = glu_module.gate_proj.weight.detach()   # (intermediate, in_dim)
    W_up = glu_module.up_proj.weight.detach()       # (intermediate, in_dim)

    # T_ijk = sum_m W_down[i,m] * W_gate[m,j] * W_up[m,k]
    T = torch.einsum('im,mj,mk->ijk', W_down, W_gate, W_up)
    return T


def analyze_tensor_spectrum(T, title=""):
    """Analyze the spectrum of the bilinear tensor.

    For each output dimension i, T[i,:,:] is a matrix.
    We can also reshape T to (out_dim, in_dim*in_dim) and do SVD.
    """
    out_dim, in_dim1, in_dim2 = T.shape

    # Reshape to 2D: (out_dim, in_dim^2) and do SVD
    T_flat = T.reshape(out_dim, -1).numpy()
    U, S, Vh = np.linalg.svd(T_flat, full_matrices=False)

    return U, S, Vh


def fourier_concentration_of_vector(v, p):
    """Compute Fourier concentration of a vector of length p."""
    spectrum = np.abs(np.fft.fft(v)) ** 2
    spectrum[0] = 0
    total = spectrum.sum()
    if total < 1e-12:
        return 0.0, 0
    dominant_freq = np.argmax(spectrum)
    concentration = float(np.max(spectrum) / total)
    return concentration, dominant_freq


# ============================================================
# 1. Modular Addition: Fourier structure in GLU weights
# ============================================================
def analyze_modadd():
    print("=" * 60)
    print("  MODULAR ADDITION: GLU Weight Decomposition")
    print("=" * 60)

    for ftype in ['glu', 'moe_glu']:
        label = 'GLU' if ftype == 'glu' else 'MoE-GLU'
        print(f"\n{label}:")

        all_weight_concs = []
        all_activation_concs = []

        for seed in SEEDS:
            path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_best.pt"
            model, ckpt = load_model(path)
            if ckpt['test_acc'] < 0.5:
                print(f"  s{seed}: not grokked, skipping")
                continue

            if ftype == 'glu':
                ffn = model.ffn
                T = get_bilinear_tensor(ffn)
                U, S, Vh = analyze_tensor_spectrum(T)

                # Check Fourier concentration of right singular vectors
                # Vh has shape (min(out,in^2), in_dim^2)
                # Reshape each row back to (in_dim, in_dim) and check Fourier structure
                in_dim = ffn.gate_proj.weight.shape[1]
                weight_concs = []
                weight_freqs = []
                for k in range(min(20, len(S))):  # top 20 singular vectors
                    v = Vh[k].reshape(in_dim, in_dim)
                    # Project onto (a+b) mod p by averaging over diagonals
                    diag_avg = np.zeros(P)
                    for s in range(P):
                        vals = []
                        for a in range(min(P, in_dim)):
                            b = (s - a) % P
                            if b < in_dim:
                                vals.append(v[a, b])
                        if vals:
                            diag_avg[s] = np.mean(vals)
                    conc, freq = fourier_concentration_of_vector(diag_avg, P)
                    weight_concs.append(conc)
                    weight_freqs.append(freq)

                mean_conc = np.mean(weight_concs[:10])
                all_weight_concs.append(mean_conc)
                top_freqs = [weight_freqs[k] for k in range(min(5, len(weight_freqs)))]
                print(f"  s{seed}: weight Fourier conc (top-10 SVs) = {mean_conc:.4f}, "
                      f"top singular values = {np.round(S[:5], 2)}, "
                      f"top freqs = {top_freqs}")

            else:
                # MoE-GLU: analyze each expert
                expert_concs = []
                for e_idx, expert in enumerate(model.ffn.experts):
                    T = get_bilinear_tensor(expert)
                    U, S, Vh = analyze_tensor_spectrum(T)

                    in_dim = expert.gate_proj.weight.shape[1]
                    weight_concs = []
                    for k in range(min(10, len(S))):
                        v = Vh[k].reshape(in_dim, in_dim)
                        diag_avg = np.zeros(P)
                        for s in range(P):
                            vals = []
                            for a in range(min(P, in_dim)):
                                b = (s - a) % P
                                if b < in_dim:
                                    vals.append(v[a, b])
                            if vals:
                                diag_avg[s] = np.mean(vals)
                        conc, _ = fourier_concentration_of_vector(diag_avg, P)
                        weight_concs.append(conc)
                    expert_concs.append(np.mean(weight_concs))

                mean_conc = np.mean(expert_concs)
                all_weight_concs.append(mean_conc)
                print(f"  s{seed}: per-expert weight Fourier conc = {[f'{c:.4f}' for c in expert_concs]}, "
                      f"mean = {mean_conc:.4f}")

        if all_weight_concs:
            wc = np.array(all_weight_concs)
            print(f"\n  {label} weight Fourier conc: {wc.mean():.4f} +/- {wc.std():.4f}")
            print(f"  Compare to activation Fourier conc: {'0.071' if ftype == 'glu' else '0.176'}")
            print(f"  Compare to FFN activation Fourier conc: 0.443")


# ============================================================
# 2. Add-7: Exploratory weight analysis
# ============================================================
def analyze_add7():
    print("\n" + "=" * 60)
    print("  ADD-7: GLU Weight Decomposition (Exploratory)")
    print("=" * 60)

    for ftype in ['glu', 'moe_glu']:
        label = 'GLU' if ftype == 'glu' else 'MoE-GLU'
        print(f"\n{label}:")

        for seed in [42]:  # just one seed for exploration
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, ckpt = load_model(path)

            if ftype == 'glu':
                T = get_bilinear_tensor(model.ffn)
                U, S, Vh = analyze_tensor_spectrum(T)

                print(f"  s{seed}: tensor shape = {T.shape}")
                print(f"  Top-10 singular values: {np.round(S[:10], 3)}")
                print(f"  Singular value ratio (S[0]/S[9]): {S[0]/S[9]:.1f}")
                print(f"  Effective rank (S > 0.01*S[0]): {int((S > 0.01*S[0]).sum())}")
            else:
                for e_idx, expert in enumerate(model.ffn.experts):
                    T = get_bilinear_tensor(expert)
                    U, S, Vh = analyze_tensor_spectrum(T)
                    print(f"  s{seed} expert {e_idx}: shape={T.shape}, "
                          f"top-5 SVs={np.round(S[:5], 3)}, "
                          f"effective rank={int(int((S > 0.01*S[0]).sum()))}")


# ============================================================
# 3. Plot: Weight vs Activation Fourier concentration
# ============================================================
def plot_comparison():
    print("\n\nGenerating comparison figure...")

    # Collect weight-based Fourier concentrations
    weight_concs = {'glu': [], 'moe_glu': []}

    for ftype in ['glu', 'moe_glu']:
        for seed in SEEDS:
            path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_best.pt"
            model, ckpt = load_model(path)
            if ckpt['test_acc'] < 0.5:
                continue

            if ftype == 'glu':
                T = get_bilinear_tensor(model.ffn)
                _, S, Vh = analyze_tensor_spectrum(T)
                in_dim = model.ffn.gate_proj.weight.shape[1]
                concs = []
                for k in range(min(10, len(S))):
                    v = Vh[k].reshape(in_dim, in_dim)
                    diag_avg = np.zeros(P)
                    for s in range(P):
                        vals = []
                        for a in range(min(P, in_dim)):
                            b_val = (s - a) % P
                            if b_val < in_dim:
                                vals.append(v[a, b_val])
                        if vals:
                            diag_avg[s] = np.mean(vals)
                    conc, _ = fourier_concentration_of_vector(diag_avg, P)
                    concs.append(conc)
                weight_concs[ftype].append(np.mean(concs))
            else:
                expert_concs = []
                for expert in model.ffn.experts:
                    T = get_bilinear_tensor(expert)
                    _, S, Vh = analyze_tensor_spectrum(T)
                    in_dim = expert.gate_proj.weight.shape[1]
                    concs = []
                    for k in range(min(10, len(S))):
                        v = Vh[k].reshape(in_dim, in_dim)
                        diag_avg = np.zeros(P)
                        for s in range(P):
                            vals = []
                            for a in range(min(P, in_dim)):
                                b_val = (s - a) % P
                                if b_val < in_dim:
                                    vals.append(v[a, b_val])
                            if vals:
                                diag_avg[s] = np.mean(vals)
                        conc, _ = fourier_concentration_of_vector(diag_avg, P)
                        concs.append(conc)
                    expert_concs.append(np.mean(concs))
                weight_concs[ftype].append(np.mean(expert_concs))

    # Activation concentrations (from previous analysis)
    activation_concs = {
        'ffn': 0.443,
        'glu': 0.071,
        'moe': 0.468,
        'moe_glu': 0.176,
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ['FFN', 'GLU', 'MoE', 'MoE-GLU']
    x = np.arange(4)
    width = 0.35

    # Activation-based
    act_vals = [activation_concs['ffn'], activation_concs['glu'],
                activation_concs['moe'], activation_concs['moe_glu']]
    ax.bar(x - width/2, act_vals, width, label='Activation-based\n(per-neuron Fourier)',
           color='#ff9999', edgecolor='black', linewidth=0.5)

    # Weight-based (only GLU and MoE-GLU have tensor decomposition)
    weight_vals = [0, np.mean(weight_concs['glu']), 0, np.mean(weight_concs['moe_glu'])]
    weight_stds = [0, np.std(weight_concs['glu']), 0, np.std(weight_concs['moe_glu'])]
    bars = ax.bar(x + width/2, weight_vals, width, yerr=weight_stds,
                  label='Weight-based\n(tensor decomposition)',
                  color='#9999ff', edgecolor='black', linewidth=0.5, capsize=3)

    # Mark N/A for FFN and MoE (not bilinear)
    for i in [0, 2]:
        ax.text(i + width/2, 0.02, 'N/A', ha='center', fontsize=10, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fourier Concentration")
    ax.set_title("Activation vs Weight Fourier Structure on (a+b) mod 113\n"
                 "(Pearce et al. tensor decomposition)")
    ax.legend()
    ax.set_ylim(0, 0.7)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig("figures/fig_glu_weight_vs_activation.png")
    print("Saved fig_glu_weight_vs_activation.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    analyze_modadd()
    analyze_add7()
    plot_comparison()
    print("\nDone!")
