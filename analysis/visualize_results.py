"""
Publication-quality figures for MoE expressivity paper.
Consistent colors: FFN=blue, GLU=orange, MoE=green, MoE-GLU=red
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

sys.path.insert(0, ".")
sys.path.insert(0, "model")
from model import OneLayerTransformer
from data.modular_addition import ModularAdditionDataset

P = 113
DEVICE = "cpu"
SEEDS = [42, 137, 256, 512, 1024]
NUM_DIGITS = 3
EOS_TOKEN = 11
PAD_TOKEN = 10

COLORS = {
    'ffn': '#1f77b4',
    'glu': '#ff7f0e',
    'moe': '#2ca02c',
    'moe_glu': '#d62728',
}
LABELS = {
    'ffn': 'FFN',
    'glu': 'GLU',
    'moe': 'MoE',
    'moe_glu': 'MoE-GLU',
}

EPOCH_CHECKPOINTS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]


def num_to_reversed_digits(n, nd):
    d = []
    for _ in range(nd):
        d.append(n % 10); n //= 10
    return d


def generate_add7_examples():
    """Generate all 1000 add-7 examples with operation labels."""
    max_val = 10 ** NUM_DIGITS - 1
    ond = NUM_DIGITS + 1
    examples = []
    for n in range(max_val + 1):
        result = n + 7
        ind = num_to_reversed_digits(n, NUM_DIGITS)
        outd = num_to_reversed_digits(result, ond)
        ops = []
        carry = 0
        for t in range(ond):
            if t == 0:
                ops.append('+7'); d = ind[0] + 7; carry = d // 10
            elif t < NUM_DIGITS:
                if carry > 0:
                    ops.append('+1'); d = ind[t] + carry; carry = d // 10
                else:
                    ops.append('+0'); carry = 0
            else:
                ops.append('+1' if carry > 0 else '+0')
        examples.append({'seq': ind + [EOS_TOKEN] + outd + [EOS_TOKEN], 'ops': ops})
    return examples


def ablation_accuracy(model, x, targets, out_start):
    """Compute normal, no-attn, no-ffn accuracy."""
    with torch.no_grad():
        logits = model(x)
    normal = (logits.argmax(-1)[:, out_start:] == targets[:, out_start:]).float().mean().item()

    orig_attn = model.atn.forward
    model.atn.forward = lambda *a, _o=orig_attn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_na = model(x)
    no_attn = (logits_na.argmax(-1)[:, out_start:] == targets[:, out_start:]).float().mean().item()
    model.atn.forward = orig_attn

    orig_ffn = model.ffn.forward
    model.ffn.forward = lambda *a, _o=orig_ffn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_nf = model(x)
    no_ffn = (logits_nf.argmax(-1)[:, out_start:] == targets[:, out_start:]).float().mean().item()
    model.ffn.forward = orig_ffn

    return normal, no_attn, no_ffn


def modadd_ablation_accuracy(model, inputs, targets):
    """Compute normal, no-attn, no-ffn accuracy for modular addition."""
    with torch.no_grad():
        logits = model(inputs)
    normal = (logits[:, 2, :].argmax(-1) == targets).float().mean().item()

    orig_attn = model.atn.forward
    model.atn.forward = lambda *a, _o=orig_attn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_na = model(inputs)
    no_attn = (logits_na[:, 2, :].argmax(-1) == targets).float().mean().item()
    model.atn.forward = orig_attn

    orig_ffn = model.ffn.forward
    model.ffn.forward = lambda *a, _o=orig_ffn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_nf = model(inputs)
    no_ffn = (logits_nf[:, 2, :].argmax(-1) == targets).float().mean().item()
    model.ffn.forward = orig_ffn

    return normal, no_attn, no_ffn


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
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


def fourier_concentration(signal, p, top_k=5):
    spectrum = np.abs(np.fft.fft(signal)) ** 2
    spectrum[0] = 0
    total = spectrum.sum()
    if total < 1e-12:
        return 0.0
    return float(np.sort(spectrum)[-top_k:].sum() / total)


def neuron_fourier_stats(activations, a_np, b_np, p):
    num_neurons = activations.shape[1]
    ab_sum = (a_np + b_np) % p
    neuron_by_sum = np.zeros((p, num_neurons))
    for s in range(p):
        mask = ab_sum == s
        if mask.sum() > 0:
            neuron_by_sum[s] = activations[mask].mean(axis=0)
    dom_freqs, concentrations = [], []
    for n in range(num_neurons):
        spectrum = np.abs(np.fft.fft(neuron_by_sum[:, n])) ** 2
        spectrum[0] = 0
        total = spectrum.sum()
        if total < 1e-12:
            dom_freqs.append(0); concentrations.append(0.0)
        else:
            dom_freqs.append(int(np.argmax(spectrum)))
            concentrations.append(float(np.max(spectrum) / total))
    return np.array(dom_freqs), np.array(concentrations)


def get_neuron_activations(model, inputs, ffn_type):
    ffn_in_data = {}
    h = model.ffn.register_forward_pre_hook(lambda m, inp: ffn_in_data.update({"x": inp[0].detach()}))
    with torch.no_grad():
        model(inputs)
    h.remove()
    ffn_in = ffn_in_data["x"][:, 2, :]

    if ffn_type == "ffn":
        with torch.no_grad():
            act = model.ffn.activation(model.ffn.up_proj(ffn_in)).numpy()
        return {"ffn": act}
    elif ffn_type == "glu":
        with torch.no_grad():
            gate = model.ffn.activation(model.ffn.gate_proj(ffn_in))
            up = model.ffn.up_proj(ffn_in)
            act = (gate * up).numpy()
        return {"glu": act}
    elif ffn_type in ("moe", "moe_glu"):
        result = {}
        for e_idx, expert in enumerate(model.ffn.experts):
            with torch.no_grad():
                if hasattr(expert, "gate_proj"):
                    gate = expert.activation(expert.gate_proj(ffn_in))
                    up = expert.up_proj(ffn_in)
                    act = (gate * up).numpy()
                else:
                    act = expert.activation(expert.up_proj(ffn_in)).numpy()
            result[f"expert_{e_idx}"] = act
        return result


# ============================================================
# FIG 1: Grokking Timeline (multi-seed, shaded)
# ============================================================
def fig1_grokking_timeline():
    fig, ax = plt.subplots(figsize=(8, 5))

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        # Collect test acc at each epoch checkpoint across seeds
        all_curves = []
        for seed in SEEDS:
            epochs, accs = [0], [1.0 / P]  # start at chance
            for ep in EPOCH_CHECKPOINTS:
                path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_epoch{ep}.pt"
                if os.path.exists(path):
                    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
                    epochs.append(ep)
                    accs.append(ckpt["test_acc"])
            all_curves.append((epochs, accs))

        # Interpolate to common epoch grid
        common_epochs = [0] + EPOCH_CHECKPOINTS
        interp_accs = np.zeros((len(SEEDS), len(common_epochs)))
        for i, (eps, acs) in enumerate(all_curves):
            for j, ce in enumerate(common_epochs):
                if ce in eps:
                    interp_accs[i, j] = acs[eps.index(ce)]
                else:
                    # Use last known value
                    prev = [acs[k] for k, e in enumerate(eps) if e <= ce]
                    interp_accs[i, j] = prev[-1] if prev else 1.0 / P

        mean = interp_accs.mean(axis=0)
        std = interp_accs.std(axis=0)

        ax.plot(common_epochs, mean, '-o', color=COLORS[ftype], label=LABELS[ftype],
                linewidth=2, markersize=4)
        ax.fill_between(common_epochs, mean - std, np.minimum(mean + std, 1.0),
                        color=COLORS[ftype], alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Grokking on (a+b) mod 113 (no norm, 5 seeds)")
    ax.legend(loc='center right')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 40000)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig("figures/fig1_grokking_timeline.png")
    print("Saved fig1_grokking_timeline.png")


# ============================================================
# FIG 2: Regularization Baselines
# ============================================================
def fig2_regularization_baselines():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    variants = [
        ('FFN', 'modadd_ffn', '#1f77b4'),
        ('FFN+drop=0.1', 'modadd_ffn_drop01', '#6baed6'),
        ('FFN+drop=0.3', 'modadd_ffn_drop03', '#9ecae1'),
        ('FFN+wd=2.0', 'modadd_ffn_wd2', '#c6dbef'),
        ('MoE', 'modadd_moe', '#2ca02c'),
    ]

    names, ep99_means, ep99_stds, grok_fracs = [], [], [], []
    for name, prefix, color in variants:
        ep99s, grokked = [], 0
        for seed in SEEDS:
            try:
                ckpt = torch.load(f"checkpoints/{prefix}_s{seed}/modadd_best.pt",
                                  weights_only=False, map_location=DEVICE)
                if ckpt['test_acc'] > 0.5:
                    grokked += 1
            except:
                pass
            try:
                mc = torch.load(f"checkpoints/{prefix}_s{seed}/modadd_test99.pt",
                                weights_only=False, map_location=DEVICE)
                ep99s.append(mc['epoch'])
            except:
                pass
        names.append(name)
        grok_fracs.append(grokked / 5)
        if ep99s:
            ep99_means.append(np.mean(ep99s))
            ep99_stds.append(np.std(ep99s))
        else:
            ep99_means.append(40000)
            ep99_stds.append(0)

    colors = [c for _, _, c in variants]
    x = np.arange(len(names))

    # Epoch to 99%
    bars = ax1.bar(x, ep99_means, yerr=ep99_stds, color=colors, capsize=4, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha='right')
    ax1.set_ylabel("Epoch to 99% Test Accuracy")
    ax1.set_title("Grokking Speed")
    ax1.grid(True, alpha=0.2, axis='y')

    # Grok reliability
    ax2.bar(x, grok_fracs, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha='right')
    ax2.set_ylabel("Fraction of Seeds Grokked")
    ax2.set_title("Grokking Reliability (5 seeds)")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.2, axis='y')

    fig.suptitle("Regularization Baselines on (a+b) mod 113", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig2_regularization_baselines.png")
    print("Saved fig2_regularization_baselines.png")


# ============================================================
# FIG 3: Number of Experts
# ============================================================
def fig3_num_experts():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    expert_counts = [1, 2, 4, 8, 16]
    ep99_means, ep99_stds, grok_fracs = [], [], []

    for ne in expert_counts:
        ep99s, grokked = [], 0
        for seed in SEEDS:
            try:
                ckpt = torch.load(f"checkpoints/modadd_moe_e{ne}_s{seed}/modadd_best.pt",
                                  weights_only=False, map_location=DEVICE)
                if ckpt['test_acc'] > 0.5:
                    grokked += 1
            except:
                pass
            try:
                mc = torch.load(f"checkpoints/modadd_moe_e{ne}_s{seed}/modadd_test99.pt",
                                weights_only=False, map_location=DEVICE)
                ep99s.append(mc['epoch'])
            except:
                pass
        grok_fracs.append(grokked / 5)
        if ep99s:
            ep99_means.append(np.mean(ep99s))
            ep99_stds.append(np.std(ep99s))
        else:
            ep99_means.append(40000)
            ep99_stds.append(0)

    x = np.arange(len(expert_counts))
    xlabels = [str(e) for e in expert_counts]

    ax1.bar(x, ep99_means, yerr=ep99_stds, color='#2ca02c', capsize=4, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.set_xlabel("Number of Experts")
    ax1.set_ylabel("Epoch to 99% Test Accuracy")
    ax1.set_title("Grokking Speed vs Expert Count")
    ax1.grid(True, alpha=0.2, axis='y')
    # Annotate E=1
    ax1.annotate("E=1: has aux loss\nbut doesn't help", xy=(0, ep99_means[0]),
                 xytext=(0.5, 35000), fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    ax2.bar(x, grok_fracs, color='#2ca02c', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels)
    ax2.set_xlabel("Number of Experts")
    ax2.set_ylabel("Fraction of Seeds Grokked")
    ax2.set_title("Grokking Reliability vs Expert Count")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.2, axis='y')

    fig.suptitle("Effect of Number of Experts on (a+b) mod 113", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig3_num_experts.png")
    print("Saved fig3_num_experts.png")


# ============================================================
# FIG 4: H1 Component Ablation (Add-7, no norm)
# ============================================================
def fig4_h1_ablation():
    fig, ax = plt.subplots(figsize=(8, 5))

    examples = generate_add7_examples()
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1

    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']
    results = {f: {'normal': [], 'no_attn': [], 'no_ffn': []} for f in ftypes}

    for ftype in ftypes:
        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, ckpt = load_model(path)
            normal, no_attn, no_ffn = ablation_accuracy(model, x_input, targets, out_start)
            results[ftype]['normal'].append(normal)
            results[ftype]['no_attn'].append(no_attn)
            results[ftype]['no_ffn'].append(no_ffn)
        print(f"  Fig4: {ftype} done")

    x = np.arange(len(ftypes))
    width = 0.25

    no_attn_means = [np.mean(results[f]['no_attn']) for f in ftypes]
    no_attn_stds = [np.std(results[f]['no_attn']) for f in ftypes]
    no_ffn_means = [np.mean(results[f]['no_ffn']) for f in ftypes]
    no_ffn_stds = [np.std(results[f]['no_ffn']) for f in ftypes]
    normal_means = [np.mean(results[f]['normal']) for f in ftypes]

    ax.bar(x - width, no_attn_means, width, label='No Attention',
           color='#ff9999', edgecolor='black', linewidth=0.5,
           yerr=no_attn_stds, capsize=3)
    ax.bar(x, no_ffn_means, width, label='No FFN',
           color='#9999ff', edgecolor='black', linewidth=0.5,
           yerr=no_ffn_stds, capsize=3)
    ax.bar(x + width, normal_means, width, label='Normal',
           color='#99cc99', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in ftypes])
    ax.set_ylabel("Accuracy")
    ax.set_title("Component Ablation on Add-7 (no norm, 5 seeds)")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis='y')

    ax.annotate("FFN critical\nfor FFN/GLU", xy=(0, no_ffn_means[0]),
                xytext=(0.3, 0.35), fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate("MoE distributes\ncomputation", xy=(2, no_ffn_means[2]),
                xytext=(2.3, 0.75), fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))

    fig.tight_layout()
    fig.savefig("figures/fig4_h1_ablation.png")
    print("Saved fig4_h1_ablation.png")


# ============================================================
# FIG 5: H3 Expert-Operation Routing (Add-7, no norm)
# ============================================================
def fig5_h3_routing():
    from collections import defaultdict

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    examples = generate_add7_examples()
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1
    ops = ['+7', '+1', '+0']
    op_to_idx = {'+7': 0, '+1': 1, '+0': 2}

    # Use MoE-GLU seed 42 (cleanest specialization)
    path = "checkpoints/add7_moe_glu_nonorm_s42/best_model.pt"
    model, ckpt = load_model(path)
    ne = model.ffn.num_experts

    # Get routing
    hook_data = {}
    h = model.ffn.router.register_forward_hook(
        lambda m, i, o: hook_data.update({"out": o.detach()}))
    with torch.no_grad():
        model(x_input)
    h.remove()

    rl = hook_data["out"].view(-1, x_input.shape[1], ne)
    rp = F.softmax(rl, dim=-1).numpy()

    # Count routing by operation
    routing_counts = np.zeros((3, ne))
    for i, ex in enumerate(examples):
        for t, op in enumerate(ex['ops']):
            pos = NUM_DIGITS + t
            if pos < rp.shape[1]:
                expert = np.argmax(rp[i, pos])
                routing_counts[op_to_idx[op], expert] += 1

    routing_frac = routing_counts / routing_counts.sum(axis=1, keepdims=True)

    im1 = ax1.imshow(routing_frac, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(ne))
    ax1.set_xticklabels([f"Expert {i}" for i in range(ne)])
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(ops)
    ax1.set_title("Routing Fraction\n(MoE-GLU, seed 42)")
    ax1.set_xlabel("Expert")
    ax1.set_ylabel("Operation")
    plt.colorbar(im1, ax=ax1, label="Fraction of tokens")

    for i in range(3):
        for j in range(ne):
            val = routing_frac[i, j]
            ax1.text(j, i, f"{val:.2f}", ha='center', va='center',
                     color='white' if val > 0.5 else 'black', fontsize=11)

    # Expert ablation
    def per_op_acc(logits):
        preds = logits.argmax(-1)
        oc = defaultdict(list)
        for i, ex in enumerate(examples):
            for t, op in enumerate(ex['ops']):
                pos = out_start + t
                if pos < preds.shape[1]:
                    oc[op].append((preds[i, pos] == targets[i, pos]).item())
        return {op: np.mean(v) for op, v in oc.items()}

    with torch.no_grad():
        normal_acc = per_op_acc(model(x_input))

    abl_matrix = np.zeros((3, ne))
    for e_idx in range(ne):
        orig = model.ffn.experts[e_idx].forward
        model.ffn.experts[e_idx].forward = lambda x, _o=orig: torch.zeros_like(_o(x))
        with torch.no_grad():
            abl_acc = per_op_acc(model(x_input))
        model.ffn.experts[e_idx].forward = orig
        for oi, op in enumerate(ops):
            abl_matrix[oi, e_idx] = normal_acc[op] - abl_acc[op]

    im2 = ax2.imshow(abl_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.8)
    ax2.set_xticks(range(ne))
    ax2.set_xticklabels([f"Expert {i}" for i in range(ne)])
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(ops)
    ax2.set_title("Accuracy Drop When Ablated\n(MoE-GLU, seed 42)")
    ax2.set_xlabel("Expert")
    ax2.set_ylabel("Operation")
    plt.colorbar(im2, ax=ax2, label="Accuracy drop")

    for i in range(3):
        for j in range(ne):
            val = abl_matrix[i, j]
            ax2.text(j, i, f"{val:.2f}", ha='center', va='center',
                     color='white' if val > 0.4 else 'black', fontsize=11)

    fig.suptitle("Expert Specialization by Operation Type (Add-7, no norm)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig5_h3_routing.png")
    print("Saved fig5_h3_routing.png")


# ============================================================
# FIG 6: Neuron Fourier Concentration
# ============================================================
def fig6_fourier_concentration():
    inputs, targets, a_vals, b_vals = get_all_inputs()
    a_np, b_np = a_vals.numpy(), b_vals.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, ftype in enumerate(['ffn', 'glu', 'moe', 'moe_glu']):
        model, ckpt = load_model(f"checkpoints/modadd_{ftype}_s42/modadd_best.pt")
        act_dict = get_neuron_activations(model, inputs, ftype)

        all_concs = []
        for key, act in act_dict.items():
            _, concs = neuron_fourier_stats(act, a_np, b_np, P)
            all_concs.extend(concs)
        all_concs = np.array(all_concs)

        ax = axes[idx]
        ax.hist(all_concs, bins=30, color=COLORS[ftype], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(f"{LABELS[ftype]}\nmean={all_concs.mean():.3f}")
        ax.set_xlabel("Fourier Concentration")
        ax.set_ylabel("# Neurons")
        ax.set_xlim(0, 1)

    fig.suptitle("Per-Neuron Fourier Concentration on (a+b) mod 113", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig6_fourier_concentration.png")
    print("Saved fig6_fourier_concentration.png")


# ============================================================
# FIG 7: Model Width Scaling
# ============================================================
def fig7_width_scaling():
    fig, ax = plt.subplots(figsize=(7, 5))

    widths = [64, 128, 256]
    ffn_means, ffn_stds = [], []
    moe_means, moe_stds = [], []

    for width in widths:
        for ftype, means_list, stds_list in [('ffn', ffn_means, ffn_stds),
                                              ('moe', moe_means, moe_stds)]:
            ep99s = []
            for seed in SEEDS:
                if width == 128:
                    path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_test99.pt"
                else:
                    path = f"checkpoints/modadd_{ftype}_d{width}_s{seed}/modadd_test99.pt"
                try:
                    mc = torch.load(path, weights_only=False, map_location=DEVICE)
                    ep99s.append(mc['epoch'])
                except:
                    pass
            if ep99s:
                means_list.append(np.mean(ep99s))
                stds_list.append(np.std(ep99s))
            else:
                means_list.append(40000)
                stds_list.append(0)

    x = np.arange(len(widths))
    width_bar = 0.35

    ax.bar(x - width_bar / 2, ffn_means, width_bar, yerr=ffn_stds,
           label='FFN', color=COLORS['ffn'], capsize=4, edgecolor='black', linewidth=0.5)
    ax.bar(x + width_bar / 2, moe_means, width_bar, yerr=moe_stds,
           label='MoE', color=COLORS['moe'], capsize=4, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"d={w}" for w in widths])
    ax.set_xlabel("Model Dimension")
    ax.set_ylabel("Epoch to 99% Test Accuracy")
    ax.set_title("MoE Advantage Scales with Model Width")
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # Add speedup annotations
    for i in range(len(widths)):
        if moe_means[i] > 0 and ffn_means[i] > 0:
            speedup = ffn_means[i] / moe_means[i]
            ax.text(i, max(ffn_means[i], moe_means[i]) + ffn_stds[i] + 500,
                    f"{speedup:.1f}x", ha='center', fontsize=10, fontweight='bold')

    fig.tight_layout()
    fig.savefig("figures/fig7_width_scaling.png")
    print("Saved fig7_width_scaling.png")


# ============================================================
# FIG A1: Norm vs No-Norm Grokking
# ============================================================
def figa1_norm_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']

    for label, tmpl, ax in [('No Norm', 'modadd_{ftype}_s{seed}', ax1),
                              ('With Norm', 'modadd_{ftype}_norm_s{seed}', ax2)]:
        ax.set_title(label)
        for ftype in ftypes:
            all_curves = []
            for seed in SEEDS:
                prefix = tmpl.format(ftype=ftype, seed=seed)
                epochs, accs = [0], [1.0 / P]
                for ep in EPOCH_CHECKPOINTS:
                    path = f"checkpoints/{prefix}/modadd_epoch{ep}.pt"
                    if os.path.exists(path):
                        ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
                        epochs.append(ep)
                        accs.append(ckpt["test_acc"])
                all_curves.append((epochs, accs))

            common_epochs = [0] + EPOCH_CHECKPOINTS
            interp_accs = np.zeros((len(SEEDS), len(common_epochs)))
            for i, (eps, acs) in enumerate(all_curves):
                for j, ce in enumerate(common_epochs):
                    if ce in eps:
                        interp_accs[i, j] = acs[eps.index(ce)]
                    else:
                        prev = [acs[k] for k, e in enumerate(eps) if e <= ce]
                        interp_accs[i, j] = prev[-1] if prev else 1.0 / P

            mean = interp_accs.mean(axis=0)
            std = interp_accs.std(axis=0)
            ax.plot(common_epochs, mean, '-o', color=COLORS[ftype], label=LABELS[ftype],
                    linewidth=2, markersize=3)
            ax.fill_between(common_epochs, mean - std, np.minimum(mean + std, 1.0),
                            color=COLORS[ftype], alpha=0.15)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 40000)
        ax.legend(loc='center right')
        ax.grid(True, alpha=0.2)

    fig.suptitle("Effect of RMSNorm on Grokking: (a+b) mod 113", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/figa1_norm_comparison.png")
    print("Saved figa1_norm_comparison.png")


# ============================================================
# FIG A2: Top-k Comparison
# ============================================================
def figa2_topk():
    fig, ax = plt.subplots(figsize=(7, 5))

    variants = [
        ('MoE\ntop-1', 'modadd_moe', '#2ca02c'),
        ('MoE\ntop-2', 'modadd_moe_topk2', '#98df8a'),
        ('MoE-GLU\ntop-1', 'modadd_moe_glu', '#d62728'),
        ('MoE-GLU\ntop-2', 'modadd_moe_glu_topk2', '#ff9896'),
    ]

    names, accs_mean, accs_std = [], [], []
    colors = []
    for name, prefix, color in variants:
        acc_list = []
        for seed in SEEDS:
            try:
                ckpt = torch.load(f"checkpoints/{prefix}_s{seed}/modadd_best.pt",
                                  weights_only=False, map_location=DEVICE)
                acc_list.append(ckpt['test_acc'])
            except:
                pass
        a = np.array(acc_list) if acc_list else np.array([0])
        names.append(name)
        accs_mean.append(a.mean())
        accs_std.append(a.std())
        colors.append(color)

    x = np.arange(len(names))
    ax.bar(x, accs_mean, yerr=accs_std, color=colors, capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Best Test Accuracy")
    ax.set_title("Top-k Routing: Top-2 Kills Grokking")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig("figures/figa2_topk.png")
    print("Saved figa2_topk.png")


# ============================================================
# FIG 8: H1 Component Ablation (ModAdd, no norm)
# ============================================================
def fig8_h1_ablation_modadd():
    fig, ax = plt.subplots(figsize=(8, 5))

    inputs, targets, _, _ = get_all_inputs()
    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']
    results = {f: {'normal': [], 'no_attn': [], 'no_ffn': []} for f in ftypes}

    for ftype in ftypes:
        for seed in SEEDS:
            path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_best.pt"
            model, ckpt = load_model(path)
            normal, no_attn, no_ffn = modadd_ablation_accuracy(model, inputs, targets)
            results[ftype]['normal'].append(normal)
            results[ftype]['no_attn'].append(no_attn)
            results[ftype]['no_ffn'].append(no_ffn)
        print(f"  Fig8: {ftype} done")

    x = np.arange(len(ftypes))
    width = 0.25

    no_attn_means = [np.mean(results[f]['no_attn']) for f in ftypes]
    no_attn_stds = [np.std(results[f]['no_attn']) for f in ftypes]
    no_ffn_means = [np.mean(results[f]['no_ffn']) for f in ftypes]
    no_ffn_stds = [np.std(results[f]['no_ffn']) for f in ftypes]
    normal_means = [np.mean(results[f]['normal']) for f in ftypes]

    ax.bar(x - width, no_attn_means, width, label='No Attention',
           color='#ff9999', edgecolor='black', linewidth=0.5,
           yerr=no_attn_stds, capsize=3)
    ax.bar(x, no_ffn_means, width, label='No FFN',
           color='#9999ff', edgecolor='black', linewidth=0.5,
           yerr=no_ffn_stds, capsize=3)
    ax.bar(x + width, normal_means, width, label='Normal',
           color='#99cc99', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in ftypes])
    ax.set_ylabel("Accuracy")
    ax.set_title("Component Ablation on (a+b) mod 113 (no norm, 5 seeds)")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis='y')

    ax.annotate("Both components\ncritical (~1% without either)", xy=(1, no_ffn_means[1]),
                xytext=(1.5, 0.4), fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))

    fig.tight_layout()
    fig.savefig("figures/fig8_h1_ablation_modadd.png")
    print("Saved fig8_h1_ablation_modadd.png")


# ============================================================
# FIG 9: Fourier Structure Over Training
# ============================================================
def fig9_fourier_over_training():
    inputs, targets, a_vals, b_vals = get_all_inputs()
    a_np, b_np = a_vals.numpy(), b_vals.numpy()
    ab_sum = (a_np + b_np) % P

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        epochs = []
        neuron_concs = []
        router_concs = []

        for ep in EPOCH_CHECKPOINTS:
            path = f"checkpoints/modadd_{ftype}_s42/modadd_epoch{ep}.pt"
            if not os.path.exists(path):
                continue
            model, ckpt = load_model(path)
            ffn_type = ckpt["config"]["ffn_type"]
            epochs.append(ep)

            # Neuron Fourier concentration
            act_dict = get_neuron_activations(model, inputs, ffn_type)
            all_concs = []
            for key, act in act_dict.items():
                _, concs = neuron_fourier_stats(act, a_np, b_np, P)
                all_concs.extend(concs)
            neuron_concs.append(np.mean(all_concs))

            # Router Fourier concentration (MoE only)
            if ffn_type in ("moe", "moe_glu"):
                hook_data = {}
                h = model.ffn.router.register_forward_hook(
                    lambda m, i, o: hook_data.update({"out": o.detach()}))
                with torch.no_grad():
                    model(inputs)
                h.remove()
                rl = hook_data["out"].view(-1, 3, model.ffn.num_experts)
                rp = F.softmax(rl[:, 2, :], dim=-1).numpy()
                rconcs = []
                for e in range(rp.shape[1]):
                    prob_by_sum = np.zeros(P)
                    for s in range(P):
                        mask = ab_sum == s
                        prob_by_sum[s] = rp[mask, e].mean()
                    rconcs.append(fourier_concentration(prob_by_sum, P))
                router_concs.append(np.mean(rconcs))

        ax1.plot(epochs, neuron_concs, 'o-', color=COLORS[ftype], label=LABELS[ftype],
                 linewidth=2, markersize=5)
        if router_concs:
            ax2.plot(epochs, router_concs, 'o-', color=COLORS[ftype], label=LABELS[ftype],
                     linewidth=2, markersize=5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Neuron Fourier Concentration")
    ax1.set_title("(a) Neuron Fourier Structure Over Training")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Router Fourier Concentration")
    ax2.set_title("(b) Router Fourier Structure Over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Fourier Structure Emergence During Grokking (seed 42)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig9_fourier_over_training.png")
    print("Saved fig9_fourier_over_training.png")


# ============================================================
# FIG A3: Norm vs No-Norm Ablation on Add-7
# ============================================================
def figa3_norm_ablation_add7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    examples = generate_add7_examples()
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1

    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']

    # Compute ablations for both norm settings
    all_results = {}
    for setting, tmpl in [('norm', 'checkpoints/add7_{ftype}_s{seed}/best_model.pt'),
                           ('nonorm', 'checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt')]:
        all_results[setting] = {}
        for ftype in ftypes:
            no_attns, no_ffns = [], []
            for seed in SEEDS:
                path = tmpl.format(ftype=ftype, seed=seed)
                model, ckpt = load_model(path)
                _, no_attn, no_ffn = ablation_accuracy(model, x_input, targets, out_start)
                no_attns.append(no_attn)
                no_ffns.append(no_ffn)
            all_results[setting][ftype] = {
                'no_attn': (np.mean(no_attns), np.std(no_attns)),
                'no_ffn': (np.mean(no_ffns), np.std(no_ffns)),
            }
        print(f"  FigA3: {setting} done")

    x = np.arange(len(ftypes))
    width = 0.35

    # Panel (a): No-FFN accuracy comparison
    norm_ffn_means = [all_results['norm'][f]['no_ffn'][0] for f in ftypes]
    norm_ffn_stds = [all_results['norm'][f]['no_ffn'][1] for f in ftypes]
    nonorm_ffn_means = [all_results['nonorm'][f]['no_ffn'][0] for f in ftypes]
    nonorm_ffn_stds = [all_results['nonorm'][f]['no_ffn'][1] for f in ftypes]

    ax1.bar(x - width / 2, norm_ffn_means, width,
            label='With Norm', color=[COLORS[f] for f in ftypes], alpha=0.6,
            edgecolor='black', linewidth=0.5, yerr=norm_ffn_stds, capsize=3)
    ax1.bar(x + width / 2, nonorm_ffn_means, width,
            label='No Norm', color=[COLORS[f] for f in ftypes], alpha=1.0,
            edgecolor='black', linewidth=0.5, yerr=nonorm_ffn_stds, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[f] for f in ftypes])
    ax1.set_ylabel("Accuracy Without FFN")
    ax1.set_title("(a) Accuracy After FFN Ablation")
    ax1.legend()
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.annotate(f"{norm_ffn_means[2]:.0%} with norm\n{nonorm_ffn_means[2]:.0%} without",
                 xy=(2 - width / 2, norm_ffn_means[2]),
                 xytext=(2.5, 0.85), fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    # Panel (b): No-Attn accuracy comparison
    norm_attn_means = [all_results['norm'][f]['no_attn'][0] for f in ftypes]
    norm_attn_stds = [all_results['norm'][f]['no_attn'][1] for f in ftypes]
    nonorm_attn_means = [all_results['nonorm'][f]['no_attn'][0] for f in ftypes]
    nonorm_attn_stds = [all_results['nonorm'][f]['no_attn'][1] for f in ftypes]

    ax2.bar(x - width / 2, norm_attn_means, width,
            label='With Norm', color=[COLORS[f] for f in ftypes], alpha=0.6,
            edgecolor='black', linewidth=0.5, yerr=norm_attn_stds, capsize=3)
    ax2.bar(x + width / 2, nonorm_attn_means, width,
            label='No Norm', color=[COLORS[f] for f in ftypes], alpha=1.0,
            edgecolor='black', linewidth=0.5, yerr=nonorm_attn_stds, capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[f] for f in ftypes])
    ax2.set_ylabel("Accuracy Without Attention")
    ax2.set_title("(b) Accuracy After Attention Ablation")
    ax2.legend()
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.2, axis='y')

    fig.suptitle("Effect of RMSNorm on Component Ablation (Add-7, 5 seeds)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/figa3_norm_ablation_add7.png")
    print("Saved figa3_norm_ablation_add7.png")


# ============================================================
# FIG A4: Per-Seed Expert Routing Variability (Add-7, no norm)
# ============================================================
def figa4_perseed_routing():
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    ops = ['+7', '+1', '+0']
    op_to_idx = {'+7': 0, '+1': 1, '+0': 2}

    examples = generate_add7_examples()
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]

    for row, ftype in enumerate(['moe', 'moe_glu']):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, ckpt = load_model(path)

            hook_data = {}
            h = model.ffn.router.register_forward_hook(
                lambda m, i, o: hook_data.update({"out": o.detach()}))
            with torch.no_grad():
                model(x_input)
            h.remove()

            ne = model.ffn.num_experts
            rl = hook_data["out"].view(-1, x_input.shape[1], ne)
            rp = F.softmax(rl, dim=-1).numpy()

            # Compute routing fractions per operation
            routing_frac = np.zeros((3, ne))
            for i, ex in enumerate(examples):
                for t, op in enumerate(ex['ops']):
                    pos = NUM_DIGITS + t
                    if pos < rp.shape[1]:
                        expert = np.argmax(rp[i, pos])
                        routing_frac[op_to_idx[op], expert] += 1

            # Normalize rows
            for i in range(3):
                total = routing_frac[i].sum()
                if total > 0:
                    routing_frac[i] /= total

            im = ax.imshow(routing_frac, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(ne))
            ax.set_xticklabels([f"E{i}" for i in range(ne)], fontsize=9)
            ax.set_yticks(range(3))
            ax.set_yticklabels(ops if col == 0 else [], fontsize=10)

            for i in range(3):
                for j in range(ne):
                    val = routing_frac[i, j]
                    if val > 0.01:
                        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                                color='white' if val > 0.5 else 'black', fontsize=8)

            ax.set_title(f"seed={seed}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{LABELS[ftype]}\nOperation", fontsize=11)

    fig.suptitle("Expert Routing by Operation Type Across Seeds (Add-7, no norm)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/figa4_perseed_routing.png")
    print("Saved figa4_perseed_routing.png")


# ============================================================
# FIG 10: Per-Position Accuracy Under FFN Ablation (Add-7, no norm)
# ============================================================
def fig10_per_position_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    examples = generate_add7_examples()
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1
    out_len = NUM_DIGITS + 1  # ones, tens, hundreds, overflow
    pos_labels = ['Ones\n(+7)', 'Tens', 'Hundreds', 'Overflow']

    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']

    # Panel (a): Per-position accuracy with FFN zeroed out
    for ftype in ftypes:
        pos_accs = [[] for _ in range(out_len)]
        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, ckpt = load_model(path)

            orig_ffn = model.ffn.forward
            model.ffn.forward = lambda *a, _o=orig_ffn, **k: torch.zeros_like(_o(*a, **k))
            with torch.no_grad():
                logits = model(x_input)
            preds = logits.argmax(-1)
            model.ffn.forward = orig_ffn

            for t in range(out_len):
                pos = out_start + t
                if pos < preds.shape[1]:
                    acc = (preds[:, pos] == targets[:, pos]).float().mean().item()
                    pos_accs[t].append(acc)

        means = [np.mean(pa) for pa in pos_accs]
        stds = [np.std(pa) for pa in pos_accs]
        x = np.arange(out_len)
        ax1.errorbar(x, means, yerr=stds, fmt='o-', color=COLORS[ftype],
                     label=LABELS[ftype], linewidth=2, markersize=6, capsize=4)
        print(f"  Fig10a: {ftype} done")

    ax1.set_xticks(range(out_len))
    ax1.set_xticklabels(pos_labels)
    ax1.set_xlabel("Output Position")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("(a) Per-Position Accuracy Without FFN")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.2)

    # Panel (b): Per-position accuracy with attention zeroed out
    for ftype in ftypes:
        pos_accs = [[] for _ in range(out_len)]
        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, ckpt = load_model(path)

            orig_attn = model.atn.forward
            model.atn.forward = lambda *a, _o=orig_attn, **k: torch.zeros_like(_o(*a, **k))
            with torch.no_grad():
                logits = model(x_input)
            preds = logits.argmax(-1)
            model.atn.forward = orig_attn

            for t in range(out_len):
                pos = out_start + t
                if pos < preds.shape[1]:
                    acc = (preds[:, pos] == targets[:, pos]).float().mean().item()
                    pos_accs[t].append(acc)

        means = [np.mean(pa) for pa in pos_accs]
        stds = [np.std(pa) for pa in pos_accs]
        x = np.arange(out_len)
        ax2.errorbar(x, means, yerr=stds, fmt='o-', color=COLORS[ftype],
                     label=LABELS[ftype], linewidth=2, markersize=6, capsize=4)
        print(f"  Fig10b: {ftype} done")

    ax2.set_xticks(range(out_len))
    ax2.set_xticklabels(pos_labels)
    ax2.set_xlabel("Output Position")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("(b) Per-Position Accuracy Without Attention")
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Per-Position Accuracy Under Component Ablation (Add-7, no norm, 5 seeds)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig10_per_position_ablation.png")
    print("Saved fig10_per_position_ablation.png")


# ============================================================
# FIG 11: Attention Pattern Heatmaps — All 4 variants (Add-7, no norm)
# ============================================================
def fig11_attention_patterns():
    pos_labels = ['d0', 'd1', 'd2', 'EOS', 'o0', 'o1', 'o2', 'o3']
    test_numbers = [123, 456, 193, 993, 500, 295, 399, 100]

    def get_attn_weights(model, n):
        ind = num_to_reversed_digits(n, NUM_DIGITS)
        outd = num_to_reversed_digits(n + 7, NUM_DIGITS + 1)
        seq = ind + [EOS_TOKEN] + outd + [EOS_TOKEN]
        x = torch.tensor([seq[:-1]], dtype=torch.long)

        pos = torch.arange(x.shape[1])
        emb = model.vocab(x) + model.pos_embed(pos)
        normed = model.atn_norm(emb)

        B, T, D = normed.shape
        H = model.atn.num_heads
        d_head = D // H

        q = model.atn.q_proj(normed).view(B, T, H, d_head).transpose(1, 2)
        k = model.atn.k_proj(normed).view(B, T, H, d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
        if model.atn.is_causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))

        return torch.softmax(scores, dim=-1)[0].detach().numpy()  # (H, T, T)

    def plot_variant_grid(variant_paths, title, filename):
        """Plot 4 variants x 4 heads attention grid."""
        fig, axes = plt.subplots(len(variant_paths), 4, figsize=(20, 5 * len(variant_paths)))

        for row, (model_name, path) in enumerate(variant_paths):
            model, _ = load_model(path)
            num_heads = model.atn.num_heads

            all_attn = []
            for n in test_numbers:
                all_attn.append(get_attn_weights(model, n))
            avg_attn = np.mean(all_attn, axis=0)  # (H, T, T)

            for h in range(num_heads):
                ax = axes[row, h]
                im = ax.imshow(avg_attn[h], cmap='Blues', vmin=0, vmax=1, aspect='equal')
                ax.set_xticks(range(len(pos_labels)))
                ax.set_xticklabels(pos_labels, fontsize=9, rotation=45)
                ax.set_yticks(range(len(pos_labels)))
                ax.set_yticklabels(pos_labels if h == 0 else [], fontsize=9)

                if row == 0:
                    ax.set_title(f"Head {h}", fontsize=12)
                if h == 0:
                    ax.set_ylabel(f"{model_name}\n(query)", fontsize=12)

                for i in range(len(pos_labels)):
                    for j in range(len(pos_labels)):
                        val = avg_attn[h, i, j]
                        if val > 0.3:
                            ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                                    color='white' if val > 0.6 else 'black', fontsize=8)

        fig.suptitle(title, fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(f"figures/{filename}")
        print(f"Saved {filename}")

    # Main figure: all 4 variants, no norm
    nonorm_paths = [
        ('FFN', 'checkpoints/add7_ffn_nonorm_s42/best_model.pt'),
        ('GLU', 'checkpoints/add7_glu_nonorm_s42/best_model.pt'),
        ('MoE', 'checkpoints/add7_moe_nonorm_s42/best_model.pt'),
        ('MoE-GLU', 'checkpoints/add7_moe_glu_nonorm_s42/best_model.pt'),
    ]
    plot_variant_grid(nonorm_paths,
                      "Per-Head Attention Patterns (Add-7, no norm, avg over 8 examples)",
                      "fig11_attention_patterns.png")

    # Appendix figure: all 4 variants, with norm
    norm_paths = [
        ('FFN', 'checkpoints/add7_ffn_s42/best_model.pt'),
        ('GLU', 'checkpoints/add7_glu_s42/best_model.pt'),
        ('MoE', 'checkpoints/add7_moe_s42/best_model.pt'),
        ('MoE-GLU', 'checkpoints/add7_moe_glu_s42/best_model.pt'),
    ]
    plot_variant_grid(norm_paths,
                      "Per-Head Attention Patterns (Add-7, with norm, avg over 8 examples)",
                      "figa5_attention_patterns_norm.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    print("Generating figures...\n")

    print("--- Main Figures ---")
    fig1_grokking_timeline()
    fig2_regularization_baselines()
    fig3_num_experts()
    fig4_h1_ablation()
    fig5_h3_routing()
    fig6_fourier_concentration()
    fig7_width_scaling()
    fig8_h1_ablation_modadd()
    fig9_fourier_over_training()
    fig10_per_position_ablation()
    fig11_attention_patterns()

    print("\n--- Appendix Figures ---")
    figa1_norm_comparison()
    figa2_topk()
    figa3_norm_ablation_add7()
    figa4_perseed_routing()
    # Note: figa5 (attention patterns with norm) is generated inside fig11_attention_patterns()

    print("\nAll figures saved to figures/")
