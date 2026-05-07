"""
Head-specific ablation on add-7 task.

Zero out individual attention heads and measure per-operation accuracy drop.
Tests whether specific heads handle specific operations (e.g., Head 2 = digit-copying,
Head 3 = carry detection).
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
        carry_len = sum(1 for o in ops if o == '+1')
        seq = ind + [EOS_TOKEN] + outd + [EOS_TOKEN]
        examples.append({
            'n': n, 'seq': seq, 'ops': ops, 'carry_len': carry_len,
        })
    return examples


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = OneLayerTransformer(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def per_op_accuracy(model, x, targets, examples, out_start):
    """Compute accuracy stratified by operation type."""
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
    """Zero out one attention head's output and measure per-op accuracy.

    We hook into the attention output projection input, zero the head's
    contribution before the output projection combines them.
    """
    num_heads = model.atn.num_heads
    d_head = model.atn.d_head

    def zero_head_hook(module, input, output):
        # output is (B, T, D) after o_proj
        # We need to intervene before o_proj. Instead, we'll recompute.
        pass

    # Simpler approach: manually compute attention with one head zeroed
    def patched_attn_forward(x_q, x_kv=None, _orig=model.atn.forward, _head=head_idx):
        if x_kv is None:
            x_kv = x_q
        B, L, C = x_q.shape
        _, S, _ = x_kv.shape
        H = model.atn.num_heads
        D = model.atn.d_head

        q = model.atn.q_proj(x_q).view(B, L, H, D).transpose(1, 2)
        k = model.atn.k_proj(x_kv).view(B, S, H, D).transpose(1, 2)
        v = model.atn.v_proj(x_kv).view(B, S, H, D).transpose(1, 2)

        # Zero out the target head's query (effectively zeroing its contribution)
        q[:, _head, :, :] = 0
        v[:, _head, :, :] = 0

        import torch.nn.functional as F
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=model.atn.is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = model.atn.o_proj(y)
        return y

    # Swap in patched attention
    orig_forward = model.atn.forward
    model.atn.forward = patched_attn_forward

    result = per_op_accuracy(model, x, targets, examples, out_start)

    # Restore
    model.atn.forward = orig_forward

    return result


def run_head_ablation(model, examples):
    """Run head-specific ablation for all heads."""
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1
    num_heads = model.atn.num_heads

    # Normal accuracy
    normal = per_op_accuracy(model, x, targets, examples, out_start)

    # Ablate each head
    head_results = {}
    for h in range(num_heads):
        ablated = ablate_head(model, h, x, targets, examples, out_start)
        drops = {op: normal[op] - ablated[op] for op in normal}
        head_results[h] = {'ablated': ablated, 'drops': drops}

    return normal, head_results


def plot_head_ablation():
    """Plot head-specific ablation for all 4 variants."""
    examples = generate_all_examples()
    ops = ['+7', '+1', '+0']
    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']

    fig, axes = plt.subplots(len(ftypes), 4, figsize=(20, 4 * len(ftypes)))

    for row, ftype in enumerate(ftypes):
        # Average across seeds
        all_drops = {h: {op: [] for op in ops} for h in range(4)}

        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, _ = load_model(path)
            normal, head_results = run_head_ablation(model, examples)

            for h in range(4):
                for op in ops:
                    all_drops[h][op].append(head_results[h]['drops'].get(op, 0))

        for h in range(4):
            ax = axes[row, h]
            means = [np.mean(all_drops[h][op]) for op in ops]
            stds = [np.std(all_drops[h][op]) for op in ops]
            colors_ops = ['#e74c3c', '#f39c12', '#3498db']  # +7=red, +1=orange, +0=blue

            bars = ax.bar(range(3), means, yerr=stds, color=colors_ops,
                          capsize=4, edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(3))
            ax.set_xticklabels(ops)
            ax.set_ylim(-0.1, 1.0)
            ax.grid(True, alpha=0.2, axis='y')
            ax.axhline(y=0, color='black', linewidth=0.5)

            if row == 0:
                ax.set_title(f"Head {h}", fontsize=13)
            if h == 0:
                ax.set_ylabel(f"{LABELS[ftype]}\nAccuracy Drop", fontsize=12)

        print(f"  Head ablation: {ftype} done")

    fig.suptitle("Per-Head Ablation: Accuracy Drop by Operation (Add-7, no norm, 5 seeds)",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig("figures/fig_head_ablation.png")
    print("Saved fig_head_ablation.png")


def print_head_ablation_summary():
    """Print summary of head ablation results."""
    examples = generate_all_examples()
    ops = ['+7', '+1', '+0']
    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']

    print("=== Head-Specific Ablation Summary (Add-7, no norm, 5 seeds) ===\n")

    for ftype in ftypes:
        print(f"{LABELS[ftype]}:")
        all_drops = {h: {op: [] for op in ops} for h in range(4)}

        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, _ = load_model(path)
            normal, head_results = run_head_ablation(model, examples)

            for h in range(4):
                for op in ops:
                    all_drops[h][op].append(head_results[h]['drops'].get(op, 0))

        for h in range(4):
            drops_str = "  ".join(
                f"{op}={np.mean(all_drops[h][op]):+.3f}±{np.std(all_drops[h][op]):.3f}"
                for op in ops
            )
            print(f"  Head {h}: {drops_str}")
        print()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    print("Head-Specific Ablation on Add-7 (no norm)\n")
    print_head_ablation_summary()
    print("\nGenerating figure...")
    plot_head_ablation()
    print("\nDone!")
