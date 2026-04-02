"""
Direct Logit Attribution (DLA) on add-7 task.

Decomposes the final logits into contributions from:
  - Embedding (residual stream before attention)
  - Attention output
  - FFN output

For each output position, measures how much each component contributes
to the correct digit logit vs others. Stratified by carry-chain length.

Method:
  residual = embed + pos_embed
  after_attn = residual + attn(norm(residual))
  after_ffn = after_attn + ffn(norm(after_attn))
  logits = unembed(out_norm(after_ffn))

  Since tie_embeddings=False, logits = W_u @ after_ffn (no norm in no-norm setting)

  DLA decomposes: logits = W_u @ residual + W_u @ attn_out + W_u @ ffn_out

  For each position, we measure the logit of the correct digit attributed
  to each component.
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


def compute_dla(model, x, targets_at_pos):
    """
    Decompose logits into embed, attention, and FFN contributions.

    Returns dict with per-position attribution of correct logit to each component.
    """
    B, T = x.shape
    pos = torch.arange(T, device=x.device)

    with torch.no_grad():
        # Step through the model manually
        residual = model.vocab(x) + model.pos_embed(pos)  # (B, T, D)

        attn_input = model.atn_norm(residual)
        attn_out = model.atn(attn_input)  # (B, T, D)

        after_attn = residual + attn_out

        ffn_input = model.ffn_norm(after_attn)
        ffn_out = model.ffn(ffn_input)  # (B, T, D)

        after_ffn = after_attn + ffn_out

        final = model.out_norm(after_ffn)

        # Get unembedding matrix
        if model.tie_embeddings:
            W_u = model.vocab.weight  # (V, D)
        else:
            W_u = model.unembed.weight  # (V, D)

        # Compute logit contributions at each position
        # logits = final @ W_u^T = (residual + attn_out + ffn_out) @ W_u^T
        # (ignoring norm for no-norm models where out_norm is Identity)

        embed_logits = residual @ W_u.T  # (B, T, V)
        attn_logits = attn_out @ W_u.T   # (B, T, V)
        ffn_logits = ffn_out @ W_u.T     # (B, T, V)

    return embed_logits, attn_logits, ffn_logits


def run_dla(model, examples):
    """Run DLA across all examples, return per-position per-operation attributions."""
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1
    out_len = NUM_DIGITS + 1

    embed_logits, attn_logits, ffn_logits = compute_dla(model, x, targets)

    # For each output position, get the correct-token logit from each component
    results = {
        'embed': np.zeros((len(examples), out_len)),
        'attn': np.zeros((len(examples), out_len)),
        'ffn': np.zeros((len(examples), out_len)),
        'carry_len': np.array([e['carry_len'] for e in examples]),
        'ops': [e['ops'] for e in examples],
    }

    for t in range(out_len):
        seq_pos = NUM_DIGITS + t  # position in x
        target_pos = seq_pos + 1  # position in full sequence (shifted)
        if seq_pos >= x.shape[1]:
            continue

        correct_tokens = targets[:, seq_pos]  # (B,)

        for i in range(len(examples)):
            ct = correct_tokens[i].item()
            results['embed'][i, t] = embed_logits[i, seq_pos, ct].item()
            results['attn'][i, t] = attn_logits[i, seq_pos, ct].item()
            results['ffn'][i, t] = ffn_logits[i, seq_pos, ct].item()

    return results


def plot_dla_by_position():
    """Plot DLA attribution by output position for all 4 variants."""
    examples = generate_all_examples()
    pos_names = ['Ones\n(+7)', 'Tens', 'Hundreds', 'Overflow']
    out_len = NUM_DIGITS + 1

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, ftype in enumerate(['ffn', 'glu', 'moe', 'moe_glu']):
        ax = axes[idx]

        # Average across seeds
        all_embed = np.zeros((len(SEEDS), out_len))
        all_attn = np.zeros((len(SEEDS), out_len))
        all_ffn = np.zeros((len(SEEDS), out_len))

        for si, seed in enumerate(SEEDS):
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, _ = load_model(path)
            results = run_dla(model, examples)
            all_embed[si] = results['embed'].mean(axis=0)
            all_attn[si] = results['attn'].mean(axis=0)
            all_ffn[si] = results['ffn'].mean(axis=0)

        # Plot stacked bar
        x = np.arange(out_len)
        width = 0.6

        embed_mean = all_embed.mean(axis=0)
        attn_mean = all_attn.mean(axis=0)
        ffn_mean = all_ffn.mean(axis=0)

        ax.bar(x, embed_mean, width, label='Embedding', color='#cccccc', edgecolor='black', linewidth=0.5)
        ax.bar(x, attn_mean, width, bottom=embed_mean, label='Attention', color='#ff9999', edgecolor='black', linewidth=0.5)
        ax.bar(x, ffn_mean, width, bottom=embed_mean + attn_mean, label='FFN', color='#9999ff', edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(pos_names)
        ax.set_ylabel("Correct Logit Attribution")
        ax.set_title(LABELS[ftype])
        if idx == 0:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

        print(f"  DLA by position: {ftype} done")

    fig.suptitle("Direct Logit Attribution by Output Position (Add-7, no norm, 5 seeds)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_dla_by_position.png")
    print("Saved fig_dla_by_position.png")


def plot_dla_by_carry():
    """Plot DLA attribution stratified by carry length, FFN vs MoE."""
    examples = generate_all_examples()
    carry_lens = np.array([e['carry_len'] for e in examples])
    out_len = NUM_DIGITS + 1

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    for row, ftype in enumerate(['ffn', 'glu', 'moe', 'moe_glu']):
        for col, L in enumerate([0, 1, 2]):
            ax = axes[row, col]
            mask = carry_lens == L

            all_attn_frac = np.zeros((len(SEEDS), out_len))
            all_ffn_frac = np.zeros((len(SEEDS), out_len))

            for si, seed in enumerate(SEEDS):
                path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
                model, _ = load_model(path)
                results = run_dla(model, examples)

                # Compute fraction of correct logit from attn vs ffn
                attn_vals = results['attn'][mask]  # (n, out_len)
                ffn_vals = results['ffn'][mask]
                total = np.abs(attn_vals) + np.abs(ffn_vals) + 1e-10
                all_attn_frac[si] = (np.abs(attn_vals) / total).mean(axis=0)
                all_ffn_frac[si] = (np.abs(ffn_vals) / total).mean(axis=0)

            pos_names = ['Ones', 'Tens', 'Hund.', 'Over.']
            x = np.arange(out_len)
            width = 0.35

            attn_mean = all_attn_frac.mean(axis=0)
            attn_std = all_attn_frac.std(axis=0)
            ffn_mean = all_ffn_frac.mean(axis=0)
            ffn_std = all_ffn_frac.std(axis=0)

            ax.bar(x - width/2, attn_mean, width, yerr=attn_std,
                   label='Attention', color='#ff9999', edgecolor='black', linewidth=0.5, capsize=3)
            ax.bar(x + width/2, ffn_mean, width, yerr=ffn_std,
                   label='FFN', color='#9999ff', edgecolor='black', linewidth=0.5, capsize=3)

            ax.set_xticks(x)
            ax.set_xticklabels(pos_names)
            ax.set_ylim(0, 1)
            ax.set_title(f"L={L}")
            if col == 0:
                ax.set_ylabel(f"{LABELS[ftype]}\nFraction of |logit|")
            if row == 0 and col == 0:
                ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, axis='y')

        print(f"  DLA by carry: {ftype} done")

    fig.suptitle("Attention vs FFN Logit Attribution by Carry Length (Add-7, no norm, 5 seeds)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_dla_by_carry.png")
    print("Saved fig_dla_by_carry.png")


def print_dla_summary():
    """Print DLA summary statistics."""
    examples = generate_all_examples()
    out_len = NUM_DIGITS + 1

    print("\n=== DLA Summary (fraction of |correct logit| from each component) ===\n")

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        print(f"{LABELS[ftype]}:")
        all_fracs = {'attn': [], 'ffn': []}

        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, _ = load_model(path)
            results = run_dla(model, examples)

            total = np.abs(results['attn']) + np.abs(results['ffn']) + 1e-10
            attn_frac = (np.abs(results['attn']) / total).mean()
            ffn_frac = (np.abs(results['ffn']) / total).mean()
            all_fracs['attn'].append(attn_frac)
            all_fracs['ffn'].append(ffn_frac)

        af = np.array(all_fracs['attn'])
        ff = np.array(all_fracs['ffn'])
        print(f"  Attention: {af.mean():.3f} +/- {af.std():.3f}")
        print(f"  FFN:       {ff.mean():.3f} +/- {ff.std():.3f}")
        print()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    print("DLA Analysis on Add-7 (no norm)\n")
    print_dla_summary()
    print("\nGenerating figures...")
    plot_dla_by_position()
    plot_dla_by_carry()
    print("\nDone!")
