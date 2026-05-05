"""
Stratify add-7 analysis by carry-chain length L.

L=0: no carry (e.g., 123+7=130, ones overflows but tens absorbs)
L=1: one carry (e.g., 193+7=200)
L=2: two carries (e.g., 993+7=1000)
L=3: three carries (e.g., 999+7=1006) — only 1 example with 3 digits

Analyses:
1. Attention patterns stratified by L (do heads change behavior with carry?)
2. Component ablation accuracy stratified by L
3. Per-position accuracy under ablation stratified by L
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

NUM_DIGITS = 3
EOS_TOKEN = 11
DEVICE = "cpu"
SEEDS = [42, 137, 256, 512, 1024]

COLORS = {
    'ffn': '#1f77b4', 'glu': '#ff7f0e',
    'moe': '#2ca02c', 'moe_glu': '#d62728',
}
LABELS = {
    'ffn': 'FFN', 'glu': 'GLU',
    'moe': 'MoE', 'moe_glu': 'MoE-GLU',
}


def num_to_reversed_digits(n, nd):
    d = []
    for _ in range(nd):
        d.append(n % 10); n //= 10
    return d


def generate_all_examples():
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
        carry_len = sum(1 for o in ops if o == '+1')
        seq = ind + [EOS_TOKEN] + outd + [EOS_TOKEN]
        examples.append({
            'n': n, 'seq': seq, 'ops': ops, 'carry_len': carry_len,
            'in_digits': ind, 'out_digits': outd,
        })
    return examples


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = OneLayerTransformer(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


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


# ============================================================
# 1. Attention patterns stratified by carry length
# ============================================================
def plot_attention_by_carry():
    examples = generate_all_examples()
    by_carry = {0: [], 1: [], 2: []}
    for ex in examples:
        if ex['carry_len'] in by_carry:
            by_carry[ex['carry_len']].append(ex['n'])

    pos_labels = ['d0', 'd1', 'd2', 'EOS', 'o0', 'o1', 'o2', 'o3']
    carry_labels = ['L=0 (no carry)', 'L=1 (one carry)', 'L=2 (two carries)']

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        model, _ = load_model(f"checkpoints/add7_{ftype}_nonorm_s42/best_model.pt")
        num_heads = model.atn.num_heads

        fig, axes = plt.subplots(3, num_heads, figsize=(20, 15))

        for row, (L, label) in enumerate(zip([0, 1, 2], carry_labels)):
            nums = by_carry[L][:50]  # sample up to 50 examples per L
            all_attn = [get_attn_weights(model, n) for n in nums]
            avg_attn = np.mean(all_attn, axis=0)  # (H, T, T)

            for h in range(num_heads):
                ax = axes[row, h]
                im = ax.imshow(avg_attn[h], cmap='Blues', vmin=0, vmax=1, aspect='equal')
                ax.set_xticks(range(len(pos_labels)))
                ax.set_xticklabels(pos_labels, fontsize=8, rotation=45)
                ax.set_yticks(range(len(pos_labels)))
                ax.set_yticklabels(pos_labels if h == 0 else [], fontsize=8)

                if row == 0:
                    ax.set_title(f"Head {h}", fontsize=12)
                if h == 0:
                    ax.set_ylabel(f"{label}\n(query)", fontsize=11)

                for i in range(len(pos_labels)):
                    for j in range(len(pos_labels)):
                        val = avg_attn[h, i, j]
                        if val > 0.3:
                            ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                                    color='white' if val > 0.6 else 'black', fontsize=7)

        fig.suptitle(f"{LABELS[ftype]} Attention Patterns by Carry Length (Add-7, no norm, seed 42)",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(f"figures/fig_attn_by_carry_{ftype}.png")
        print(f"Saved fig_attn_by_carry_{ftype}.png")


# ============================================================
# 2. Component ablation accuracy stratified by carry length
# ============================================================
def plot_ablation_by_carry():
    examples = generate_all_examples()
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1

    carry_lens = np.array([e['carry_len'] for e in examples])

    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    conditions = ['Normal', 'No Attention', 'No FFN']

    for ftype in ftypes:
        # Average across seeds
        results_by_L = {L: {'normal': [], 'no_attn': [], 'no_ffn': []} for L in [0, 1, 2]}

        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, _ = load_model(path)

            # Normal
            with torch.no_grad():
                logits = model(x_input)
            preds = logits.argmax(-1)

            # No attention
            orig_attn = model.atn.forward
            model.atn.forward = lambda *a, _o=orig_attn, **k: torch.zeros_like(_o(*a, **k))
            with torch.no_grad():
                preds_na = model(x_input).argmax(-1)
            model.atn.forward = orig_attn

            # No FFN
            orig_ffn = model.ffn.forward
            model.ffn.forward = lambda *a, _o=orig_ffn, **k: torch.zeros_like(_o(*a, **k))
            with torch.no_grad():
                preds_nf = model(x_input).argmax(-1)
            model.ffn.forward = orig_ffn

            for L in [0, 1, 2]:
                mask = carry_lens == L
                if mask.sum() == 0:
                    continue
                normal_acc = (preds[mask, out_start:] == targets[mask, out_start:]).float().mean().item()
                na_acc = (preds_na[mask, out_start:] == targets[mask, out_start:]).float().mean().item()
                nf_acc = (preds_nf[mask, out_start:] == targets[mask, out_start:]).float().mean().item()
                results_by_L[L]['normal'].append(normal_acc)
                results_by_L[L]['no_attn'].append(na_acc)
                results_by_L[L]['no_ffn'].append(nf_acc)

        for ci, cond in enumerate(['normal', 'no_attn', 'no_ffn']):
            means = [np.mean(results_by_L[L][cond]) for L in [0, 1, 2]]
            stds = [np.std(results_by_L[L][cond]) for L in [0, 1, 2]]
            axes[ci].errorbar([0, 1, 2], means, yerr=stds, fmt='o-',
                              color=COLORS[ftype], label=LABELS[ftype],
                              linewidth=2, markersize=6, capsize=4)

        print(f"  Ablation by carry: {ftype} done")

    for ci, title in enumerate(conditions):
        axes[ci].set_xticks([0, 1, 2])
        axes[ci].set_xticklabels(['L=0', 'L=1', 'L=2'])
        axes[ci].set_xlabel("Carry Chain Length")
        axes[ci].set_ylabel("Accuracy")
        axes[ci].set_title(title)
        axes[ci].legend()
        axes[ci].set_ylim(-0.05, 1.05)
        axes[ci].grid(True, alpha=0.2)

    fig.suptitle("Component Ablation by Carry Chain Length (Add-7, no norm, 5 seeds)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_ablation_by_carry.png")
    print("Saved fig_ablation_by_carry.png")


# ============================================================
# 3. Per-position accuracy under ablation, stratified by L
# ============================================================
def plot_per_position_by_carry():
    examples = generate_all_examples()
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1
    out_len = NUM_DIGITS + 1
    pos_names = ['Ones\n(+7)', 'Tens', 'Hundreds', 'Overflow']

    carry_lens = np.array([e['carry_len'] for e in examples])

    # All 4 variants, no-FFN ablation, by carry length
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        for L, ax in zip([0, 1, 2], axes):
            mask = carry_lens == L
            if mask.sum() == 0:
                continue

            pos_accs = [[] for _ in range(out_len)]
            for seed in SEEDS:
                path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
                model, _ = load_model(path)

                orig_ffn = model.ffn.forward
                model.ffn.forward = lambda *a, _o=orig_ffn, **k: torch.zeros_like(_o(*a, **k))
                with torch.no_grad():
                    preds = model(x_input).argmax(-1)
                model.ffn.forward = orig_ffn

                for t in range(out_len):
                    pos = out_start + t
                    if pos < preds.shape[1]:
                        acc = (preds[mask, pos] == targets[mask, pos]).float().mean().item()
                        pos_accs[t].append(acc)

            means = [np.mean(pa) for pa in pos_accs]
            stds = [np.std(pa) for pa in pos_accs]
            ax.errorbar(range(out_len), means, yerr=stds, fmt='o-',
                        color=COLORS[ftype], label=LABELS[ftype],
                        linewidth=2, markersize=6, capsize=4)

    for i, ax in enumerate(axes):
        ax.set_xticks(range(out_len))
        ax.set_xticklabels(pos_names)
        ax.set_xlabel("Output Position")
        ax.set_ylabel("Accuracy (no FFN)")
        ax.set_title(f"L={i}")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Per-Position Accuracy Without FFN by Carry Length (Add-7, no norm, 5 seeds)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_perpos_by_carry.png")
    print("Saved fig_perpos_by_carry.png")


# ============================================================
# Print summary statistics
# ============================================================
def print_carry_distribution():
    examples = generate_all_examples()
    from collections import Counter
    counts = Counter(e['carry_len'] for e in examples)
    print("Carry length distribution:")
    for L in sorted(counts.keys()):
        print(f"  L={L}: {counts[L]} examples ({100*counts[L]/len(examples):.1f}%)")
    print()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    print_carry_distribution()
    print("1. Attention patterns by carry length...")
    plot_attention_by_carry()
    print("\n2. Component ablation by carry length...")
    plot_ablation_by_carry()
    print("\n3. Per-position accuracy by carry length...")
    plot_per_position_by_carry()
    print("\nDone!")
