"""
Head-specific ablation with functional sorting.

Instead of averaging by head index (which varies across seeds),
sort heads by their primary function for each seed, then average.

For each seed: identify which head causes the biggest drop for each operation,
label them by function ("+7 head", "+1 head", etc.), then aggregate.
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
from collections import defaultdict

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
        seq = ind + [EOS_TOKEN] + outd + [EOS_TOKEN]
        examples.append({'n': n, 'seq': seq, 'ops': ops})
    return examples


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = OneLayerTransformer(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def per_op_accuracy(model, x, targets, examples, out_start):
    with torch.no_grad():
        logits = model(x)
    preds = logits.argmax(dim=-1)
    op_correct = defaultdict(list)
    for i, ex in enumerate(examples):
        for t, op in enumerate(ex['ops']):
            pos = out_start + t
            if pos < preds.shape[1]:
                op_correct[op].append((preds[i, pos] == targets[i, pos]).item())
    return {op: np.mean(vals) for op, vals in op_correct.items()}


def ablate_head(model, head_idx, x, targets, examples, out_start):
    def patched_attn_forward(x_q, x_kv=None, _head=head_idx):
        if x_kv is None:
            x_kv = x_q
        B, L, C = x_q.shape
        _, S, _ = x_kv.shape
        H = model.atn.num_heads
        D = model.atn.d_head

        q = model.atn.q_proj(x_q).view(B, L, H, D).transpose(1, 2)
        k = model.atn.k_proj(x_kv).view(B, S, H, D).transpose(1, 2)
        v = model.atn.v_proj(x_kv).view(B, S, H, D).transpose(1, 2)

        q[:, _head, :, :] = 0
        v[:, _head, :, :] = 0

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=model.atn.is_causal)
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = model.atn.o_proj(y)
        return y

    orig_forward = model.atn.forward
    model.atn.forward = patched_attn_forward
    result = per_op_accuracy(model, x, targets, examples, out_start)
    model.atn.forward = orig_forward
    return result


def get_head_drops(model, examples):
    """Get per-head accuracy drops for all operations."""
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1
    num_heads = model.atn.num_heads

    normal = per_op_accuracy(model, x, targets, examples, out_start)
    drops = {}
    for h in range(num_heads):
        ablated = ablate_head(model, h, x, targets, examples, out_start)
        drops[h] = {op: normal[op] - ablated[op] for op in normal}
    return normal, drops


def sort_heads_by_function(drops):
    """Sort heads by primary function.

    Returns a mapping: functional_role -> (head_index, drops_dict)
    Roles: "most +7", "most +1", "most +0", "least critical"
    """
    ops = ['+7', '+1', '+0']
    num_heads = len(drops)

    # Compute total drop per head
    total_drops = {h: sum(drops[h][op] for op in ops) for h in range(num_heads)}

    # Find the head most critical for each operation
    assigned = set()
    roles = {}

    # Assign by largest unique drop per operation
    for op in ops:
        # Sort heads by drop for this operation (descending)
        sorted_heads = sorted(range(num_heads), key=lambda h: drops[h][op], reverse=True)
        for h in sorted_heads:
            if h not in assigned:
                roles[f"Most {op}"] = h
                assigned.add(h)
                break

    # Remaining head is "least critical"
    for h in range(num_heads):
        if h not in assigned:
            roles["Least critical"] = h
            break

    return roles


def run_sorted_analysis():
    """Run head ablation with functional sorting across all variants and seeds."""
    examples = generate_all_examples()
    ops = ['+7', '+1', '+0']
    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']
    role_names = ["Most +7", "Most +1", "Most +0", "Least critical"]

    print("=== Head Ablation with Functional Sorting ===\n")

    all_variant_results = {}

    for ftype in ftypes:
        print(f"{LABELS[ftype]}:")
        sorted_drops = {role: {op: [] for op in ops} for role in role_names}
        role_head_indices = {role: [] for role in role_names}

        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, _ = load_model(path)
            normal, drops = get_head_drops(model, examples)
            roles = sort_heads_by_function(drops)

            for role_name in role_names:
                h = roles[role_name]
                role_head_indices[role_name].append(h)
                for op in ops:
                    sorted_drops[role_name][op].append(drops[h][op])

        for role_name in role_names:
            indices = role_head_indices[role_name]
            drops_str = "  ".join(
                f"{op}={np.mean(sorted_drops[role_name][op]):+.3f}±{np.std(sorted_drops[role_name][op]):.3f}"
                for op in ops
            )
            print(f"  {role_name:>16} (heads={indices}): {drops_str}")

        all_variant_results[ftype] = sorted_drops
        print()

    return all_variant_results


def plot_sorted_head_ablation(all_results):
    """Plot head ablation sorted by functional role."""
    ops = ['+7', '+1', '+0']
    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']
    role_names = ["Most +7", "Most +1", "Most +0", "Least critical"]
    colors_ops = ['#e74c3c', '#f39c12', '#3498db']

    fig, axes = plt.subplots(len(ftypes), 4, figsize=(20, 4 * len(ftypes)))

    for row, ftype in enumerate(ftypes):
        for col, role in enumerate(role_names):
            ax = axes[row, col]
            means = [np.mean(all_results[ftype][role][op]) for op in ops]
            stds = [np.std(all_results[ftype][role][op]) for op in ops]

            ax.bar(range(3), means, yerr=stds, color=colors_ops,
                   capsize=4, edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(3))
            ax.set_xticklabels(ops)
            ax.set_ylim(-0.1, 1.0)
            ax.grid(True, alpha=0.2, axis='y')
            ax.axhline(y=0, color='black', linewidth=0.5)

            if row == 0:
                ax.set_title(role, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{LABELS[ftype]}\nAccuracy Drop", fontsize=12)

    fig.suptitle("Head Ablation Sorted by Function (Add-7, no norm, 5 seeds)",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig("figures/fig_head_ablation_sorted.png")
    print("Saved fig_head_ablation_sorted.png")


def plot_comparison_bar():
    """Simpler comparison: for each variant, show the max drop per operation across heads."""
    examples = generate_all_examples()
    ops = ['+7', '+1', '+0']
    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(ftypes))
    width = 0.25
    colors_ops = ['#e74c3c', '#f39c12', '#3498db']

    for oi, op in enumerate(ops):
        max_drops_mean = []
        max_drops_std = []
        for ftype in ftypes:
            seed_max_drops = []
            for seed in SEEDS:
                path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
                model, _ = load_model(path)
                _, drops = get_head_drops(model, examples)
                # Max drop across heads for this operation
                max_drop = max(drops[h][op] for h in range(4))
                seed_max_drops.append(max_drop)
            max_drops_mean.append(np.mean(seed_max_drops))
            max_drops_std.append(np.std(seed_max_drops))

        ax.bar(x + oi * width, max_drops_mean, width, yerr=max_drops_std,
               label=op, color=colors_ops[oi], capsize=4, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([LABELS[f] for f in ftypes])
    ax.set_ylabel("Max Accuracy Drop (most critical head)")
    ax.set_title("Most Critical Head's Impact per Operation (Add-7, no norm, 5 seeds)")
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig("figures/fig_head_max_drop.png")
    print("Saved fig_head_max_drop.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    print("Head-Specific Ablation with Functional Sorting\n")
    all_results = run_sorted_analysis()

    print("Generating figures...")
    plot_sorted_head_ablation(all_results)
    plot_comparison_bar()
    print("\nDone!")
