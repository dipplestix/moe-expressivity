"""
GLU Gate Probe Analysis.

Probe the gate activations, up projections, and their product separately
to explain *how* GLU hides internal structure.

Hypothesis: Information is split between gate and up projections.
Neither alone reveals the full picture, but the product does.

We test on:
1. Modular addition — probe for Fourier structure in gate vs up vs product
2. Add-7 — probe for operation type (+7/+1/+0) from gate vs up vs product
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ".")
sys.path.insert(0, "model")

import numpy as np
import torch
import torch.nn.functional as F
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
from data.modular_addition import ModularAdditionDataset

P = 113
DEVICE = "cpu"
SEEDS = [42, 137, 256, 512, 1024]
NUM_DIGITS = 3
EOS_TOKEN = 11


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = OneLayerTransformer(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def get_glu_components(model, inputs, pos_idx=None):
    """Extract gate, up, and product activations from a GLU model.

    pos_idx: which sequence position to extract (None = all positions flattened)
    """
    ffn_in_data = {}
    h = model.ffn.register_forward_pre_hook(
        lambda m, inp: ffn_in_data.update({"x": inp[0].detach()}))
    with torch.no_grad():
        model(inputs)
    h.remove()

    ffn_in = ffn_in_data["x"]
    if pos_idx is not None:
        ffn_in = ffn_in[:, pos_idx, :]

    ffn = model.ffn
    with torch.no_grad():
        gate_pre = ffn.gate_proj(ffn_in)
        gate_post = ffn.activation(gate_pre)
        up = ffn.up_proj(ffn_in)
        product = gate_post * up

    return {
        'gate': gate_post.numpy(),
        'up': up.numpy(),
        'product': product.numpy(),
    }


def get_moe_glu_components(model, inputs, pos_idx=None):
    """Extract gate, up, product from all experts in MoE-GLU, applied to all inputs."""
    ffn_in_data = {}
    h = model.ffn.register_forward_pre_hook(
        lambda m, inp: ffn_in_data.update({"x": inp[0].detach()}))
    with torch.no_grad():
        model(inputs)
    h.remove()

    ffn_in = ffn_in_data["x"]
    if pos_idx is not None:
        ffn_in = ffn_in[:, pos_idx, :]

    # Average across experts
    all_gate, all_up, all_product = [], [], []
    for expert in model.ffn.experts:
        with torch.no_grad():
            gate_post = expert.activation(expert.gate_proj(ffn_in))
            up = expert.up_proj(ffn_in)
            product = gate_post * up
        all_gate.append(gate_post.numpy())
        all_up.append(up.numpy())
        all_product.append(product.numpy())

    return {
        'gate': np.concatenate(all_gate, axis=-1),
        'up': np.concatenate(all_up, axis=-1),
        'product': np.concatenate(all_product, axis=-1),
    }


# ============================================================
# 1. Fourier concentration in gate vs up vs product (ModAdd)
# ============================================================
def analyze_fourier_glu():
    """Compare Fourier concentration in gate, up, and product for GLU on modadd."""
    inputs_a = torch.arange(P).repeat_interleave(P)
    inputs_b = torch.arange(P).repeat(P)
    eq = torch.full_like(inputs_a, P)
    inputs = torch.stack([inputs_a, inputs_b, eq], dim=1)
    a_np, b_np = inputs_a.numpy(), inputs_b.numpy()
    ab_sum = (a_np + b_np) % P

    print("=== Fourier Concentration: Gate vs Up vs Product (ModAdd) ===\n")

    results = {}
    for ftype in ['glu', 'moe_glu']:
        label = 'GLU' if ftype == 'glu' else 'MoE-GLU'
        all_concs = {'gate': [], 'up': [], 'product': []}

        for seed in SEEDS:
            path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_best.pt"
            model, ckpt = load_model(path)
            if ckpt['test_acc'] < 0.5:
                continue

            if ftype == 'glu':
                comps = get_glu_components(model, inputs, pos_idx=2)
            else:
                comps = get_moe_glu_components(model, inputs, pos_idx=2)

            for comp_name, act in comps.items():
                num_neurons = act.shape[1]
                concs = []
                for n in range(num_neurons):
                    neuron_by_sum = np.zeros(P)
                    for s in range(P):
                        mask = ab_sum == s
                        neuron_by_sum[s] = act[mask, n].mean()
                    spectrum = np.abs(np.fft.fft(neuron_by_sum)) ** 2
                    spectrum[0] = 0
                    total = spectrum.sum()
                    if total > 1e-12:
                        concs.append(float(np.max(spectrum) / total))
                    else:
                        concs.append(0.0)
                all_concs[comp_name].append(np.mean(concs))

        for comp_name in ['gate', 'up', 'product']:
            vals = np.array(all_concs[comp_name])
            print(f"  {label} {comp_name:>8}: {vals.mean():.4f} +/- {vals.std():.4f}")
        results[ftype] = all_concs
        print()

    return results


# ============================================================
# 2. Operation type probe accuracy: gate vs up vs product (Add-7)
# ============================================================
def analyze_probes_glu():
    """Train linear probes to predict operation type from gate, up, product on add-7."""
    def num_to_reversed_digits(n, nd):
        d = []
        for _ in range(nd):
            d.append(n % 10); n //= 10
        return d

    examples = []
    for n in range(1000):
        result = n + 7
        ind = num_to_reversed_digits(n, NUM_DIGITS)
        outd = num_to_reversed_digits(result, NUM_DIGITS + 1)
        ops = []
        carry = 0
        for t in range(NUM_DIGITS + 1):
            if t == 0:
                ops.append('+7'); d = ind[0] + 7; carry = d // 10
            elif t < NUM_DIGITS:
                if carry > 0:
                    ops.append('+1'); d = ind[t] + carry; carry = d // 10
                else:
                    ops.append('+0'); carry = 0
            else:
                ops.append('+1' if carry > 0 else '+0')
        seq = ind + [EOS_TOKEN] + outd + [EOS_TOKEN]
        examples.append({'seq': seq, 'ops': ops})

    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x = seqs[:, :-1]
    op_to_idx = {'+7': 0, '+1': 1, '+0': 2}

    print("=== Operation Probe Accuracy: Gate vs Up vs Product (Add-7) ===\n")

    results = {}
    for ftype in ['glu', 'moe_glu']:
        label = 'GLU' if ftype == 'glu' else 'MoE-GLU'
        all_accs = {'gate': [], 'up': [], 'product': []}

        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, ckpt = load_model(path)

            # Get FFN input
            ffn_in_data = {}
            h = model.ffn.register_forward_pre_hook(
                lambda m, inp: ffn_in_data.update({"x": inp[0].detach()}))
            with torch.no_grad():
                model(x)
            h.remove()

            ffn_in = ffn_in_data["x"]

            # Get components
            ffn = model.ffn
            if ftype == 'glu':
                with torch.no_grad():
                    gate = ffn.activation(ffn.gate_proj(ffn_in))
                    up = ffn.up_proj(ffn_in)
                    product = gate * up
                comps = {'gate': gate, 'up': up, 'product': product}
            else:
                # For MoE-GLU, use all experts on all inputs
                all_g, all_u, all_p = [], [], []
                for expert in ffn.experts:
                    with torch.no_grad():
                        g = expert.activation(expert.gate_proj(ffn_in))
                        u = expert.up_proj(ffn_in)
                        p = g * u
                    all_g.append(g); all_u.append(u); all_p.append(p)
                comps = {
                    'gate': torch.cat(all_g, dim=-1),
                    'up': torch.cat(all_u, dim=-1),
                    'product': torch.cat(all_p, dim=-1),
                }

            # Collect vectors and labels at output positions
            out_start = NUM_DIGITS
            vecs = {k: [] for k in comps}
            labels = []
            for i, ex in enumerate(examples):
                for t, op in enumerate(ex['ops']):
                    pos = out_start + t
                    if pos < comps['gate'].shape[1]:
                        for k in comps:
                            vecs[k].append(comps[k][i, pos])
                        labels.append(op_to_idx[op])

            y = torch.tensor(labels)

            for comp_name in ['gate', 'up', 'product']:
                X = torch.stack(vecs[comp_name]).detach()
                Xm = X.mean(0); Xs = X.std(0).clamp(min=1e-6)
                Xn = (X - Xm) / Xs

                probe = torch.nn.Linear(Xn.shape[1], 3)
                opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
                for _ in range(500):
                    loss = F.cross_entropy(probe(Xn), y)
                    opt.zero_grad(); loss.backward(); opt.step()

                with torch.no_grad():
                    acc = (probe(Xn).argmax(-1) == y).float().mean().item()
                all_accs[comp_name].append(acc)

        for comp_name in ['gate', 'up', 'product']:
            vals = np.array(all_accs[comp_name])
            print(f"  {label} {comp_name:>8}: {vals.mean():.4f} +/- {vals.std():.4f}")
        results[ftype] = all_accs
        print()

    return results


# ============================================================
# 3. Plot results
# ============================================================
def plot_results(fourier_results, probe_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Fourier concentration
    comp_names = ['gate', 'up', 'product']
    x = np.arange(len(comp_names))
    width = 0.35
    colors = ['#ff7f0e', '#d62728']  # GLU orange, MoE-GLU red

    for i, (ftype, color) in enumerate(zip(['glu', 'moe_glu'], colors)):
        if ftype in fourier_results:
            means = [np.mean(fourier_results[ftype][c]) for c in comp_names]
            stds = [np.std(fourier_results[ftype][c]) for c in comp_names]
            label = 'GLU' if ftype == 'glu' else 'MoE-GLU'
            ax1.bar(x + i * width, means, width, yerr=stds, label=label,
                    color=color, capsize=4, edgecolor='black', linewidth=0.5)

    # Add FFN reference line
    ax1.axhline(y=0.44, color='#1f77b4', linestyle='--', alpha=0.7, label='FFN (0.44)')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(['Gate\nσ(W_g·x)', 'Up\nW_u·x', 'Product\ngate × up'])
    ax1.set_ylabel("Mean Fourier Concentration")
    ax1.set_title("(a) Fourier Structure in GLU Components\n(Modular Addition)")
    ax1.legend()
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_ylim(0, 0.6)

    # Panel 2: Probe accuracy
    for i, (ftype, color) in enumerate(zip(['glu', 'moe_glu'], colors)):
        if ftype in probe_results:
            means = [np.mean(probe_results[ftype][c]) for c in comp_names]
            stds = [np.std(probe_results[ftype][c]) for c in comp_names]
            label = 'GLU' if ftype == 'glu' else 'MoE-GLU'
            ax2.bar(x + i * width, means, width, yerr=stds, label=label,
                    color=color, capsize=4, edgecolor='black', linewidth=0.5)

    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(['Gate\nσ(W_g·x)', 'Up\nW_u·x', 'Product\ngate × up'])
    ax2.set_ylabel("Probe Accuracy (op type)")
    ax2.set_title("(b) Operation Type Probe on GLU Components\n(Add-7, no norm)")
    ax2.legend()
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_ylim(0.5, 1.05)

    fig.suptitle("GLU Gate Decomposition: Where Is the Information?", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_glu_gate_probes.png")
    print("Saved fig_glu_gate_probes.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    fourier_results = analyze_fourier_glu()
    probe_results = analyze_probes_glu()
    plot_results(fourier_results, probe_results)
    print("\nDone!")
