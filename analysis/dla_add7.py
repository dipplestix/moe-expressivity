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
os.chdir("<PATH_TO_REPO>")
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
    cfg = dict(ckpt['config'])
    # Recover intermediate_dim from state_dict if missing (e.g., narrow FFN checkpoints).
    sd = ckpt['model_state_dict']
    if 'intermediate_dim' not in cfg:
        if cfg.get('ffn_type') == 'ffn' and 'ffn.up_proj.weight' in sd:
            cfg['intermediate_dim'] = sd['ffn.up_proj.weight'].shape[0]
        elif cfg.get('ffn_type') == 'glu' and 'ffn.up_proj.weight' in sd:
            cfg['intermediate_dim'] = sd['ffn.up_proj.weight'].shape[0]
        elif cfg.get('ffn_type') == 'moe' and 'ffn.experts.0.up_proj.weight' in sd:
            cfg['intermediate_dim'] = sd['ffn.experts.0.up_proj.weight'].shape[0] * cfg.get('num_experts', 4)
        elif cfg.get('ffn_type') == 'moe_glu' and 'ffn.experts.0.up_proj.weight' in sd:
            cfg['intermediate_dim'] = sd['ffn.experts.0.up_proj.weight'].shape[0] * cfg.get('num_experts', 4)
    # Recover random_routing flag from checkpoint metadata if present
    if 'random_routing' in ckpt and 'random_routing' not in cfg:
        cfg['random_routing'] = ckpt['random_routing']
    model = OneLayerTransformer(**cfg)
    model.load_state_dict(sd)
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


def compute_dla_per_head(model, x):
    """
    Decompose attention DLA by head.

    Replicates MHA forward, splits y by head, and projects each head's
    contribution to the residual stream through W_U.

    Per head: contribution_h = y_h @ W_O[h*D:(h+1)*D, :].T  (shape (B, T, d_model))
    Per-head logits: contribution_h @ W_U.T  (shape (B, T, V))

    Excludes shared o_proj bias (a constant per-token logit shift, not
    informative about per-head specialization).
    """
    B, T = x.shape
    pos = torch.arange(T, device=x.device)

    with torch.no_grad():
        residual = model.vocab(x) + model.pos_embed(pos)
        attn_input = model.atn_norm(residual)

        atn = model.atn
        H = atn.num_heads
        D = atn.d_head

        q = atn._shape(atn.q_proj(attn_input), B, T)
        k = atn._shape(atn.k_proj(attn_input), B, T)
        v = atn._shape(atn.v_proj(attn_input), B, T)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=atn.is_causal)
        # y: (B, H, T, D)

        if model.tie_embeddings:
            W_u = model.vocab.weight
        else:
            W_u = model.unembed.weight

        # o_proj: out = y_merged @ W_O.T + b. W_O has shape (d_model, d_model).
        # Per-head row block: W_O[:, h*D:(h+1)*D] is the slice that multiplies head h.
        # Equivalently y_h @ W_O[:, h*D:(h+1)*D].T gives head h's residual contribution.
        W_O = atn.o_proj.weight  # (d_model, d_model)

        per_head_logits = torch.zeros(B, T, H, W_u.shape[0], device=x.device)
        for h in range(H):
            y_h = y[:, h, :, :]  # (B, T, D)
            W_O_h = W_O[:, h * D:(h + 1) * D]  # (d_model, D)
            contribution_h = y_h @ W_O_h.T  # (B, T, d_model)
            per_head_logits[:, :, h, :] = contribution_h @ W_u.T

    return per_head_logits


def run_dla_per_head(model, examples):
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_len = NUM_DIGITS + 1
    H = model.num_heads

    per_head_logits = compute_dla_per_head(model, x)

    results = {
        'per_head': np.zeros((len(examples), out_len, H)),
        'carry_len': np.array([e['carry_len'] for e in examples]),
    }

    for t in range(out_len):
        seq_pos = NUM_DIGITS + t
        if seq_pos >= x.shape[1]:
            continue
        correct_tokens = targets[:, seq_pos]
        for i in range(len(examples)):
            ct = correct_tokens[i].item()
            for h in range(H):
                results['per_head'][i, t, h] = per_head_logits[i, seq_pos, h, ct].item()

    return results


def plot_per_head_dla():
    """Per-head attention DLA by output position, across architectures.

    Heads are sorted within each seed by total |correct-logit contribution|
    summed over output positions, so rank 1 = largest contributor in that seed.
    This avoids averaging over permutation-arbitrary head indices.
    """
    examples = generate_all_examples()
    pos_names = ['Ones\n(+7)', 'Tens', 'Hund.', 'Over.']
    out_len = NUM_DIGITS + 1

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    rank_colors = ['#d62728', '#ff9896', '#aec7e8', '#1f77b4', '#2ca02c', '#9467bd']

    for idx, ftype in enumerate(['ffn', 'glu', 'moe', 'moe_glu']):
        ax = axes[idx]

        H = None
        all_per_head = None

        for si, seed in enumerate(SEEDS):
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, _ = load_model(path)
            if H is None:
                H = model.num_heads
                all_per_head = np.zeros((len(SEEDS), out_len, H))
            results = run_dla_per_head(model, examples)
            all_per_head[si] = results['per_head'].mean(axis=0)

        # Sort heads within each seed by total |contribution| across positions
        totals = np.abs(all_per_head).sum(axis=1)  # (S, H)
        sort_idx = np.argsort(-totals, axis=1)  # (S, H), descending
        sorted_ph = np.zeros_like(all_per_head)
        for si in range(len(SEEDS)):
            sorted_ph[si] = all_per_head[si][:, sort_idx[si]]

        x_base = np.arange(out_len)
        width = 0.8 / H

        for r in range(H):
            mean = sorted_ph[:, :, r].mean(axis=0)
            std = sorted_ph[:, :, r].std(axis=0)
            offset = (r - (H - 1) / 2) * width
            ax.bar(x_base + offset, mean, width, yerr=std,
                   label=f'Rank {r + 1}', color=rank_colors[r % len(rank_colors)],
                   edgecolor='black', linewidth=0.5, capsize=2)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x_base)
        ax.set_xticklabels(pos_names)
        if idx == 0:
            ax.set_ylabel("Direct logit contribution")
        ax.set_title(LABELS[ftype])
        if idx == 0:
            ax.legend(fontsize=9, loc='best', title='Head rank\n(within seed)')
        ax.grid(True, alpha=0.2, axis='y')

        print(f"  Per-head DLA: {ftype} done")

    fig.suptitle("Per-Head Attention DLA by Output Position (heads sorted within seed)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_dla_per_head.png")
    print("Saved fig_dla_per_head.png")


def plot_per_head_dla_causal():
    """Per-head DLA causal decomposition on add-7.

    Compares four conditions that align with the redistribution decomposition:
      - dense FFN          (baseline)
      - narrow FFN         (reduced per-token capacity only)
      - random-routing MoE (sparse partitioning, no learned routing)
      - learned MoE        (full MoE)

    Top row: per-head signed contribution by output position (sorted within seed).
    Bottom: Rank-1 share = |rank 1| / sum_r |rank r|, by output position, all
    conditions on one axis. Bounded in [0, 1]; the headline concentration metric.
    """
    examples = generate_all_examples()
    pos_names = ['Ones\n(+7)', 'Tens', 'Hund.', 'Over.']
    out_len = NUM_DIGITS + 1

    variants = [
        ('FFN',               '#1f77b4', lambda s: f"checkpoints/add7_ffn_nonorm_s{s}/best_model.pt"),
        ('Narrow FFN',        '#9467bd', lambda s: f"checkpoints/add7_ffn_narrow_nonorm_s{s}/best_model.pt"),
        ('Random-route MoE',  '#2ca02c', lambda s: f"checkpoints/add7_moe_randroute_nonorm_s{s}/best_model.pt"),
        ('Learned MoE',       '#d62728', lambda s: f"checkpoints/add7_moe_nonorm_s{s}/best_model.pt"),
    ]

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.8, 1], hspace=0.5, wspace=0.1)
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax in top_axes[1:]:
        ax.sharey(top_axes[0])
    bottom_ax = fig.add_subplot(gs[1, :])

    rank_colors = ['#d62728', '#ff9896', '#aec7e8', '#1f77b4', '#2ca02c', '#9467bd']
    share_per_condition = []

    for idx, (label, color, path_fn) in enumerate(variants):
        ax = top_axes[idx]
        H = None
        all_per_head = None

        for si, seed in enumerate(SEEDS):
            model, _ = load_model(path_fn(seed))
            if H is None:
                H = model.num_heads
                all_per_head = np.zeros((len(SEEDS), out_len, H))
            results = run_dla_per_head(model, examples)
            all_per_head[si] = results['per_head'].mean(axis=0)

        totals = np.abs(all_per_head).sum(axis=1)
        sort_idx = np.argsort(-totals, axis=1)
        sorted_ph = np.zeros_like(all_per_head)
        for si in range(len(SEEDS)):
            sorted_ph[si] = all_per_head[si][:, sort_idx[si]]

        # Top panel: per-head signed contribution
        x_base = np.arange(out_len)
        width = 0.8 / H
        for r in range(H):
            mean = sorted_ph[:, :, r].mean(axis=0)
            std = sorted_ph[:, :, r].std(axis=0)
            offset = (r - (H - 1) / 2) * width
            ax.bar(x_base + offset, mean, width, yerr=std,
                   label=f'Rank {r + 1}', color=rank_colors[r % len(rank_colors)],
                   edgecolor='black', linewidth=0.5, capsize=2)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x_base)
        ax.set_xticklabels(pos_names, fontsize=10)
        if idx == 0:
            ax.set_ylabel("Signed direct\nlogit contribution", fontsize=11)
        ax.set_title(label, color=color, fontsize=12)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper left', title='Head rank\n(within seed)', title_fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')
        ax.tick_params(axis='y', labelsize=10)

        # Compute Rank-1 share per seed per position
        abs_sorted = np.abs(sorted_ph)  # (S, out_len, H)
        denom = abs_sorted.sum(axis=2) + 1e-10  # (S, out_len)
        rank1_share = abs_sorted[:, :, 0] / denom  # (S, out_len)
        share_per_condition.append((label, color, rank1_share.mean(axis=0), rank1_share.std(axis=0)))

        print(f"  Causal per-head DLA: {label} done")

    # Bottom panel: Rank-1 share comparison
    n_cond = len(variants)
    cond_width = 0.8 / n_cond
    x_base = np.arange(out_len)
    for ci, (label, color, mean, std) in enumerate(share_per_condition):
        offset = (ci - (n_cond - 1) / 2) * cond_width
        bottom_ax.bar(x_base + offset, mean, cond_width, yerr=std,
                      label=label, color=color, edgecolor='black', linewidth=0.5, capsize=3)
    bottom_ax.axhline(1.0 / 4, color='black', linewidth=0.8, linestyle=':',
                      label='Uniform $1/H$')
    bottom_ax.set_xticks(x_base)
    bottom_ax.set_xticklabels(['Ones (+7)', 'Tens', 'Hund.', 'Over.'], fontsize=11)
    bottom_ax.set_ylabel("Rank-1 share", fontsize=11)
    bottom_ax.set_ylim(0, 1)
    bottom_ax.set_title(r"Rank-1 share $= |\ell_{\mathrm{rank}\,1}| / \sum_r |\ell_{\mathrm{rank}\,r}|$",
                        fontsize=11)
    bottom_ax.legend(fontsize=9, loc='upper left', ncol=5, frameon=True)
    bottom_ax.grid(True, alpha=0.2, axis='y')
    bottom_ax.tick_params(axis='y', labelsize=10)

    fig.suptitle("Per-head attention DLA on add-7 (heads sorted within seed, 5 seeds)",
                 fontsize=13, y=0.995)
    fig.savefig("figures/fig_dla_per_head_causal.png")
    print("Saved fig_dla_per_head_causal.png")

    # Print Rank-1 share summary
    print("\n=== Rank-1 share (|rank 1| / sum_r |rank r|), carry positions average ===")
    for label, _, mean, std in share_per_condition:
        carry_mean = mean[1:].mean()  # Tens, Hund., Over.
        print(f"  {label:20s}  carry-mean Rank-1 share = {carry_mean:.3f}")


def print_per_head_causal_summary():
    """Summary table for the four causal conditions."""
    examples = generate_all_examples()
    out_len = NUM_DIGITS + 1
    variants = [
        ('FFN',               lambda s: f"checkpoints/add7_ffn_nonorm_s{s}/best_model.pt"),
        ('Narrow FFN',        lambda s: f"checkpoints/add7_ffn_narrow_nonorm_s{s}/best_model.pt"),
        ('Random-route MoE',  lambda s: f"checkpoints/add7_moe_randroute_nonorm_s{s}/best_model.pt"),
        ('Learned MoE',       lambda s: f"checkpoints/add7_moe_nonorm_s{s}/best_model.pt"),
    ]
    print("\n=== Per-Head DLA — Causal Decomposition (heads sorted within seed) ===\n")
    for label, path_fn in variants:
        print(f"{label}:")
        seed_means = []
        H = None
        for seed in SEEDS:
            model, _ = load_model(path_fn(seed))
            if H is None:
                H = model.num_heads
            results = run_dla_per_head(model, examples)
            seed_means.append(results['per_head'].mean(axis=0))

        arr = np.array(seed_means)
        totals = np.abs(arr).sum(axis=1)
        sort_idx = np.argsort(-totals, axis=1)
        arr_sorted = np.zeros_like(arr)
        for si in range(arr.shape[0]):
            arr_sorted[si] = arr[si][:, sort_idx[si]]
        mean = arr_sorted.mean(axis=0)
        std = arr_sorted.std(axis=0)

        pos_names = ['Ones', 'Tens', 'Hund.', 'Over.']
        header = "  Position | " + " | ".join([f"  Rank {r + 1}" for r in range(H)])
        print(header)
        print("  " + "-" * (len(header) - 2))
        for t in range(out_len):
            row = f"  {pos_names[t]:8s} | " + " | ".join([f"{mean[t, r]:+6.3f}±{std[t, r]:.2f}" for r in range(H)])
            print(row)
        # Concentration ratio at carry positions
        carry_positions = [1, 2, 3]  # Tens, Hund., Over.
        rank1_carry = mean[carry_positions, 0].mean()
        rank4_carry = mean[carry_positions, -1].mean()
        print(f"  Carry-position avg: Rank 1 = {rank1_carry:+.3f}, Rank {H} = {rank4_carry:+.3f}, "
              f"ratio = {rank1_carry / (abs(rank4_carry) + 1e-3):+.1f}")
        print()


def print_per_head_summary():
    """Per-head DLA summary with heads sorted within seed (rank 1 = largest)."""
    examples = generate_all_examples()
    out_len = NUM_DIGITS + 1

    print("\n=== Per-Head Attention DLA (heads sorted within seed by total |contribution|) ===\n")

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        print(f"{LABELS[ftype]}:")
        seed_means = []
        H = None

        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, _ = load_model(path)
            if H is None:
                H = model.num_heads
            results = run_dla_per_head(model, examples)
            seed_means.append(results['per_head'].mean(axis=0))  # (out_len, H)

        arr = np.array(seed_means)  # (S, out_len, H)
        totals = np.abs(arr).sum(axis=1)  # (S, H)
        sort_idx = np.argsort(-totals, axis=1)
        arr_sorted = np.zeros_like(arr)
        for si in range(arr.shape[0]):
            arr_sorted[si] = arr[si][:, sort_idx[si]]

        mean = arr_sorted.mean(axis=0)
        std = arr_sorted.std(axis=0)

        pos_names = ['Ones', 'Tens', 'Hund.', 'Over.']
        header = "  Position | " + " | ".join([f"  Rank {r + 1}" for r in range(H)])
        print(header)
        print("  " + "-" * (len(header) - 2))
        for t in range(out_len):
            row = f"  {pos_names[t]:8s} | " + " | ".join([f"{mean[t, r]:+6.3f}±{std[t, r]:.2f}" for r in range(H)])
            print(row)
        print()


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
    print_per_head_summary()
    print("\nGenerating figures...")
    plot_dla_by_position()
    plot_dla_by_carry()
    plot_per_head_dla()
    print("\nDone!")
