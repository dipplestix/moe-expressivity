"""
Activation patching on add-7 task.

Swap attention or FFN outputs between examples with different operations
at the same position to test causal role of each component.

Method:
  1. Pick pairs of examples that differ in operation at a target position
     (e.g., position 1: +0 in one, +1 in the other)
  2. Run both through the model, capture intermediate activations
  3. Patch activation from example A into example B at the target position
  4. Measure if model B now produces example A's output at that position
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ".")
sys.path.insert(0, "model")

import numpy as np
import torch
from collections import defaultdict
from model import OneLayerTransformer

NUM_DIGITS = 3
EOS_TOKEN = 11
DEVICE = "cpu"
SEEDS = [42, 137, 256, 512, 1024]


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
        examples.append({
            'n': n, 'seq': seq, 'ops': ops,
            'in_digits': ind, 'out_digits': outd,
        })
    return examples


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = OneLayerTransformer(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def get_activations(model, x):
    """Forward pass capturing attention and FFN outputs."""
    hooks = {}
    handles = []

    handles.append(model.atn.register_forward_hook(
        lambda m, i, o: hooks.update({'attn_out': o.detach().clone()})))
    handles.append(model.ffn.register_forward_hook(
        lambda m, i, o: hooks.update({'ffn_out': o.detach().clone()})))

    with torch.no_grad():
        logits = model(x)

    for h in handles:
        h.remove()

    return logits, hooks


def patched_forward(model, x, patch_component, patch_pos, patch_value):
    """Forward pass with one activation patched at a specific position.

    patch_component: 'attn' or 'ffn'
    patch_pos: which sequence position to patch
    patch_value: the activation tensor to insert (1, D)
    """
    def make_attn_hook(pos, val):
        def hook_fn(module, input, output):
            patched = output.clone()
            patched[:, pos, :] = val
            return patched
        return hook_fn

    def make_ffn_hook(pos, val):
        def hook_fn(module, input, output):
            patched = output.clone()
            patched[:, pos, :] = val
            return patched
        return hook_fn

    if patch_component == 'attn':
        h = model.atn.register_forward_hook(make_attn_hook(patch_pos, patch_value))
    else:
        h = model.ffn.register_forward_hook(make_ffn_hook(patch_pos, patch_value))

    with torch.no_grad():
        logits = model(x)

    h.remove()
    return logits


def find_pairs(examples, target_output_pos):
    """Find pairs of examples that differ in operation at target_output_pos.

    Returns pairs (example_with_op_A, example_with_op_B) where they have
    the same input digit at that position but different operations.
    """
    seq_pos = NUM_DIGITS + 1 + target_output_pos  # position in full sequence

    # Group by (input digit at this position, operation at this position)
    by_digit_op = defaultdict(list)
    for ex in examples:
        if target_output_pos < len(ex['ops']):
            if target_output_pos < NUM_DIGITS:
                input_digit = ex['in_digits'][target_output_pos]
            else:
                input_digit = 'overflow'
            op = ex['ops'][target_output_pos]
            by_digit_op[(input_digit, op)].append(ex)

    # Find pairs with same input digit but different operations
    pairs = []
    digits = set(d for d, _ in by_digit_op.keys())
    for digit in digits:
        ops_for_digit = [op for d, op in by_digit_op.keys() if d == digit]
        for i, op_a in enumerate(ops_for_digit):
            for op_b in ops_for_digit[i+1:]:
                exs_a = by_digit_op[(digit, op_a)]
                exs_b = by_digit_op[(digit, op_b)]
                # Sample pairs
                for ea in exs_a[:5]:
                    for eb in exs_b[:5]:
                        pairs.append((ea, eb, target_output_pos, op_a, op_b))
    return pairs


def run_patching_experiment(model, examples):
    """Run activation patching for all output positions."""
    results = {}

    for target_pos in range(NUM_DIGITS + 1):  # ones, tens, hundreds, overflow
        pairs = find_pairs(examples, target_pos)
        if not pairs:
            continue

        seq_pos = NUM_DIGITS + 1 + target_pos  # position in model input sequence
        # The target in the input sequence is at seq_pos - 1 (shifted by 1 for next-token prediction)
        target_seq_pos = NUM_DIGITS + target_pos  # position in x (input without last token)

        attn_flips = 0
        ffn_flips = 0
        total = 0

        for ex_a, ex_b, tpos, op_a, op_b in pairs:
            x_a = torch.tensor([ex_a['seq'][:-1]], dtype=torch.long)
            x_b = torch.tensor([ex_b['seq'][:-1]], dtype=torch.long)

            # Get clean outputs
            logits_a, hooks_a = get_activations(model, x_a)
            logits_b, hooks_b = get_activations(model, x_b)

            pred_b_clean = logits_b[0, target_seq_pos].argmax().item()
            expected_a = ex_a['seq'][target_seq_pos + 1]  # what A outputs at this position

            # Patch attention output from A into B
            attn_val = hooks_a['attn_out'][0, target_seq_pos]
            logits_b_patched_attn = patched_forward(model, x_b, 'attn', target_seq_pos, attn_val)
            pred_b_attn_patched = logits_b_patched_attn[0, target_seq_pos].argmax().item()

            # Patch FFN output from A into B
            ffn_val = hooks_a['ffn_out'][0, target_seq_pos]
            logits_b_patched_ffn = patched_forward(model, x_b, 'ffn', target_seq_pos, ffn_val)
            pred_b_ffn_patched = logits_b_patched_ffn[0, target_seq_pos].argmax().item()

            # Did patching flip the prediction to A's answer?
            if pred_b_attn_patched == expected_a and pred_b_clean != expected_a:
                attn_flips += 1
            if pred_b_ffn_patched == expected_a and pred_b_clean != expected_a:
                ffn_flips += 1
            total += 1

        if total > 0:
            pos_names = ['ones (+7)', 'tens', 'hundreds', 'overflow']
            results[pos_names[target_pos]] = {
                'attn_flip_rate': attn_flips / total,
                'ffn_flip_rate': ffn_flips / total,
                'total_pairs': total,
            }

    return results


if __name__ == "__main__":
    examples = generate_all_examples()
    print(f"Generated {len(examples)} examples\n")

    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']
    labels = {'ffn': 'FFN', 'glu': 'GLU', 'moe': 'MoE', 'moe_glu': 'MoE-GLU'}

    for ftype in ftypes:
        print(f"{'='*60}")
        print(f"  {labels[ftype]} (no norm)")
        print(f"{'='*60}")

        all_results = defaultdict(lambda: {'attn_flips': [], 'ffn_flips': []})

        for seed in SEEDS:
            path = f"checkpoints/add7_{ftype}_nonorm_s{seed}/best_model.pt"
            model, ckpt = load_model(path)
            results = run_patching_experiment(model, examples)

            print(f"\n  seed={seed}:")
            for pos_name, r in results.items():
                print(f"    {pos_name}: attn_flip={r['attn_flip_rate']:.3f} "
                      f"ffn_flip={r['ffn_flip_rate']:.3f} (n={r['total_pairs']})")
                all_results[pos_name]['attn_flips'].append(r['attn_flip_rate'])
                all_results[pos_name]['ffn_flips'].append(r['ffn_flip_rate'])

        print(f"\n  --- Aggregated ({len(SEEDS)} seeds) ---")
        for pos_name in ['ones (+7)', 'tens', 'hundreds', 'overflow']:
            if pos_name in all_results:
                af = np.array(all_results[pos_name]['attn_flips'])
                ff = np.array(all_results[pos_name]['ffn_flips'])
                print(f"    {pos_name}: attn_flip={af.mean():.3f}+/-{af.std():.3f} "
                      f"ffn_flip={ff.mean():.3f}+/-{ff.std():.3f}")
        print()
