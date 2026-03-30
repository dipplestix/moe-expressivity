"""
H1-H3 analysis for add-7 task across all 4 architecture variants.

H1 (Separation): Attention computes selection, FFN executes digit mapping
  - Component ablation: zero attention vs zero FFN output
  - Linear probes on attention/FFN outputs to predict operation type

H2 (GLU effect): GLU gates predict operation type more cleanly
  - Gate activation analysis by operation type (+7, +1, +0)

H3 (MoE specialization): Expert routing aligns with operation types
  - Routing patterns stratified by operation type
  - Expert ablation by operation type
  - Mutual information between expert id and operation type
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, "model")
from model import OneLayerTransformer

# Token definitions
PAD_TOKEN = 10
EOS_TOKEN = 11
VOCAB_SIZE = 12
NUM_DIGITS = 3


def num_to_reversed_digits(n, num_digits):
    digits = []
    for _ in range(num_digits):
        digits.append(n % 10)
        n //= 10
    return digits


def generate_all_examples(num_digits):
    """Generate all possible add-7 examples with operation labels."""
    max_val = 10 ** num_digits - 1
    out_num_digits = num_digits + 1

    examples = []
    for n in range(max_val + 1):
        result = n + 7
        in_digits = num_to_reversed_digits(n, num_digits)
        out_digits = num_to_reversed_digits(result, out_num_digits)

        # Determine per-position operation type
        # Position 0 (ones): always +7
        # Position t > 0: +1 if carry reaches here, +0 otherwise
        ops = []
        carry = 0
        for t in range(out_num_digits):
            if t == 0:
                ops.append('+7')
                d = in_digits[0] + 7
                carry = d // 10
            elif t < num_digits:
                if carry > 0:
                    ops.append('+1')
                    d = in_digits[t] + carry
                    carry = d // 10
                else:
                    ops.append('+0')
                    carry = 0
            else:
                # Overflow digit
                if carry > 0:
                    ops.append('+1')
                else:
                    ops.append('+0')

        seq = in_digits + [EOS_TOKEN] + out_digits + [EOS_TOKEN]
        examples.append({
            'n': n,
            'seq': seq,
            'in_digits': in_digits,
            'out_digits': out_digits,
            'ops': ops,
            'carry_len': sum(1 for o in ops if o == '+1'),
        })

    return examples


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    config = ckpt['config']
    model = OneLayerTransformer(**config).to('cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def run_with_hooks(model, x):
    """Forward pass capturing attention output, FFN input, and FFN output."""
    hooks = {}
    handles = []

    def capture(name):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                hooks[name] = out[0].detach()
            else:
                hooks[name] = out.detach()
        return hook_fn

    def capture_input(name):
        def hook_fn(module, inp, out):
            hooks[name] = inp[0].detach()
        return hook_fn

    handles.append(model.atn.register_forward_hook(capture('attn_out')))
    handles.append(model.ffn.register_forward_hook(capture('ffn_out')))
    handles.append(model.ffn.register_forward_pre_hook(
        lambda m, inp: hooks.update({'ffn_in': inp[0].detach()})
    ))

    # For MoE, capture router
    if hasattr(model.ffn, 'router'):
        handles.append(model.ffn.router.register_forward_hook(capture('router_out')))

    with torch.no_grad():
        logits = model(x)

    for h in handles:
        h.remove()

    return logits, hooks


# ============================================================
# H1: Component Ablation
# ============================================================
def h1_ablation(model, examples, ffn_type):
    """Zero out attention or FFN and measure accuracy drop."""
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1  # position after first EOS in target

    # Normal accuracy
    with torch.no_grad():
        logits = model(x)
    preds = logits.argmax(dim=-1)
    normal_acc = (preds[:, out_start:] == targets[:, out_start:]).float()

    # Per-position accuracy
    out_positions = NUM_DIGITS + 1  # number of output positions (including overflow + EOS)
    pos_accs_normal = []
    for t in range(out_positions):
        pos_accs_normal.append(normal_acc[:, t].mean().item())

    # Ablate attention (zero out attention output)
    orig_attn_forward = model.atn.forward
    def zero_attn(*args, **kwargs):
        out = orig_attn_forward(*args, **kwargs)
        return torch.zeros_like(out)
    model.atn.forward = zero_attn

    with torch.no_grad():
        logits_no_attn = model(x)
    preds_no_attn = logits_no_attn.argmax(dim=-1)
    no_attn_acc = (preds_no_attn[:, out_start:] == targets[:, out_start:]).float()
    pos_accs_no_attn = [no_attn_acc[:, t].mean().item() for t in range(out_positions)]
    model.atn.forward = orig_attn_forward

    # Ablate FFN (zero out FFN output)
    orig_ffn_forward = model.ffn.forward
    def zero_ffn(*args, **kwargs):
        out = orig_ffn_forward(*args, **kwargs)
        return torch.zeros_like(out)
    model.ffn.forward = zero_ffn

    with torch.no_grad():
        logits_no_ffn = model(x)
    preds_no_ffn = logits_no_ffn.argmax(dim=-1)
    no_ffn_acc = (preds_no_ffn[:, out_start:] == targets[:, out_start:]).float()
    pos_accs_no_ffn = [no_ffn_acc[:, t].mean().item() for t in range(out_positions)]
    model.ffn.forward = orig_ffn_forward

    return {
        'normal': pos_accs_normal,
        'no_attn': pos_accs_no_attn,
        'no_ffn': pos_accs_no_ffn,
        'overall_normal': normal_acc.mean().item(),
        'overall_no_attn': no_attn_acc.mean().item(),
        'overall_no_ffn': no_ffn_acc.mean().item(),
    }


# ============================================================
# H1: Linear Probes
# ============================================================
def h1_linear_probes(model, examples, ffn_type):
    """Train linear probes to predict operation type from attention/FFN outputs."""
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x = seqs[:, :-1]

    _, hooks = run_with_hooks(model, x)
    attn_out = hooks['attn_out']  # (N, T, D)
    ffn_out = hooks['ffn_out']    # (N, T, D)

    # Collect (activation, op_label) for output positions
    op_to_idx = {'+7': 0, '+1': 1, '+0': 2}
    out_start = NUM_DIGITS  # position of first EOS in input (which predicts first output)

    attn_vecs = []
    ffn_vecs = []
    labels = []

    for i, ex in enumerate(examples):
        for t, op in enumerate(ex['ops']):
            pos = out_start + t  # position in the sequence
            if pos < attn_out.shape[1]:
                attn_vecs.append(attn_out[i, pos])
                ffn_vecs.append(ffn_out[i, pos])
                labels.append(op_to_idx[op])

    attn_X = torch.stack(attn_vecs)  # (N_tokens, D)
    ffn_X = torch.stack(ffn_vecs)
    y = torch.tensor(labels)

    # Train simple logistic regression probe
    results = {}
    for name, X in [('attn', attn_X), ('ffn', ffn_X)]:
        # Standardize
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0).clamp(min=1e-6)
        X_norm = (X - X_mean) / X_std

        # Train probe with SGD
        probe = torch.nn.Linear(X_norm.shape[1], 3)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)

        for _ in range(500):
            logits = probe(X_norm)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            preds = probe(X_norm).argmax(dim=-1)
            acc = (preds == y).float().mean().item()

            # Per-class accuracy
            per_class = {}
            for op_name, op_idx in op_to_idx.items():
                mask = y == op_idx
                if mask.sum() > 0:
                    per_class[op_name] = (preds[mask] == y[mask]).float().mean().item()

        results[name] = {'accuracy': acc, 'per_class': per_class}

    return results


# ============================================================
# H3: MoE Routing by Operation Type
# ============================================================
def h3_routing_analysis(model, examples):
    """Analyze which experts handle which operation types."""
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x = seqs[:, :-1]

    _, hooks = run_with_hooks(model, x)

    router_logits = hooks['router_out']  # (N*T, E)
    num_experts = router_logits.shape[-1]
    T = x.shape[1]
    router_logits = router_logits.view(-1, T, num_experts)
    router_probs = F.softmax(router_logits, dim=-1)

    out_start = NUM_DIGITS
    op_to_idx = {'+7': 0, '+1': 1, '+0': 2}

    # Collect routing probs by operation type
    routing_by_op = defaultdict(list)
    expert_selection_by_op = defaultdict(list)

    for i, ex in enumerate(examples):
        for t, op in enumerate(ex['ops']):
            pos = out_start + t
            if pos < router_probs.shape[1]:
                probs = router_probs[i, pos].numpy()
                routing_by_op[op].append(probs)
                expert_selection_by_op[op].append(np.argmax(probs))

    # Compute statistics
    results = {}
    for op in ['+7', '+1', '+0']:
        probs = np.array(routing_by_op[op])
        selections = np.array(expert_selection_by_op[op])
        mean_probs = probs.mean(axis=0)
        expert_counts = np.bincount(selections, minlength=num_experts)
        results[op] = {
            'mean_probs': mean_probs,
            'expert_counts': expert_counts,
            'n_tokens': len(probs),
        }

    # Mutual information between expert selection and operation type
    # MI = H(expert) + H(op) - H(expert, op)
    all_experts = []
    all_ops = []
    for op in ['+7', '+1', '+0']:
        sels = expert_selection_by_op[op]
        all_experts.extend(sels)
        all_ops.extend([op_to_idx[op]] * len(sels))
    all_experts = np.array(all_experts)
    all_ops = np.array(all_ops)

    def entropy(x):
        counts = np.bincount(x)
        probs = counts[counts > 0] / counts.sum()
        return -np.sum(probs * np.log2(probs))

    H_expert = entropy(all_experts)
    H_op = entropy(all_ops)

    # Joint entropy
    joint = all_experts * 3 + all_ops
    H_joint = entropy(joint)
    MI = H_expert + H_op - H_joint

    results['MI'] = MI
    results['H_expert'] = H_expert
    results['H_op'] = H_op
    results['normalized_MI'] = MI / min(H_expert, H_op) if min(H_expert, H_op) > 0 else 0

    return results


# ============================================================
# H3: Expert Ablation by Operation Type
# ============================================================
def h3_expert_ablation(model, examples):
    """Ablate each expert and measure per-operation accuracy drop."""
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1
    num_experts = model.ffn.num_experts

    # Collect per-token operation labels
    op_labels = []  # (N, out_positions)
    for ex in examples:
        op_labels.append(ex['ops'])

    def compute_per_op_accuracy(logits):
        preds = logits.argmax(dim=-1)
        op_correct = defaultdict(list)
        for i, ex in enumerate(examples):
            for t, op in enumerate(ex['ops']):
                pos = out_start + t
                if pos < preds.shape[1]:
                    correct = (preds[i, pos] == targets[i, pos]).item()
                    op_correct[op].append(correct)
        return {op: np.mean(vals) for op, vals in op_correct.items()}

    # Normal accuracy
    with torch.no_grad():
        logits = model(x)
    normal_acc = compute_per_op_accuracy(logits)

    # Ablate each expert
    ablation_results = {}
    for e_idx in range(num_experts):
        orig_forward = model.ffn.experts[e_idx].forward
        model.ffn.experts[e_idx].forward = lambda x, _orig=orig_forward: torch.zeros_like(_orig(x))

        with torch.no_grad():
            logits_ablated = model(x)
        ablated_acc = compute_per_op_accuracy(logits_ablated)

        model.ffn.experts[e_idx].forward = orig_forward

        ablation_results[e_idx] = {
            op: normal_acc[op] - ablated_acc[op]
            for op in normal_acc
        }

    return {'normal': normal_acc, 'ablation_drops': ablation_results}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    examples = generate_all_examples(NUM_DIGITS)
    print(f"Generated {len(examples)} examples")

    # Count operations
    op_counts = defaultdict(int)
    for ex in examples:
        for op in ex['ops']:
            op_counts[op] += 1
    print(f"Operation counts: {dict(op_counts)}")
    print(f"Carry length distribution: ", end="")
    cl_counts = defaultdict(int)
    for ex in examples:
        cl_counts[ex['carry_len']] += 1
    print(dict(sorted(cl_counts.items())))

    seeds = [42, 137, 256, 512, 1024]

    for ftype in ['ffn', 'glu', 'moe', 'moe_glu']:
        print(f"\n{'='*60}")
        print(f"  {ftype.upper()}")
        print(f"{'='*60}")

        all_h1_ablation = []
        all_h1_probes = []
        all_h3_routing = []
        all_h3_ablation = []

        for seed in seeds:
            path = f"checkpoints/add7_{ftype}_s{seed}/best_model.pt"
            model, ckpt = load_model(path)
            print(f"\n  seed={seed}, acc={ckpt['accuracy']:.4f}, step={ckpt['step']}")

            # H1: Ablation
            h1_abl = h1_ablation(model, examples, ftype)
            all_h1_ablation.append(h1_abl)
            print(f"  H1 Ablation: normal={h1_abl['overall_normal']:.3f} "
                  f"no_attn={h1_abl['overall_no_attn']:.3f} "
                  f"no_ffn={h1_abl['overall_no_ffn']:.3f}")

            # H1: Linear probes
            h1_probes = h1_linear_probes(model, examples, ftype)
            all_h1_probes.append(h1_probes)
            print(f"  H1 Probes: attn={h1_probes['attn']['accuracy']:.3f} "
                  f"ffn={h1_probes['ffn']['accuracy']:.3f}")
            print(f"    attn per-op: {h1_probes['attn']['per_class']}")
            print(f"    ffn  per-op: {h1_probes['ffn']['per_class']}")

            # H3: Routing analysis (MoE only)
            if ftype in ('moe', 'moe_glu'):
                h3_rout = h3_routing_analysis(model, examples)
                all_h3_routing.append(h3_rout)
                print(f"  H3 Routing MI: {h3_rout['MI']:.3f} "
                      f"(normalized: {h3_rout['normalized_MI']:.3f})")
                for op in ['+7', '+1', '+0']:
                    r = h3_rout[op]
                    print(f"    {op}: mean_probs={np.round(r['mean_probs'], 3)} "
                          f"counts={r['expert_counts']}")

                # H3: Expert ablation
                h3_abl = h3_expert_ablation(model, examples)
                all_h3_ablation.append(h3_abl)
                print(f"  H3 Expert ablation (acc drops):")
                for e_idx, drops in h3_abl['ablation_drops'].items():
                    print(f"    Expert {e_idx}: "
                          + " ".join(f"{op}={d:+.3f}" for op, d in drops.items()))

        # Aggregate across seeds
        print(f"\n  --- Aggregated ({len(seeds)} seeds) ---")

        # H1 ablation means
        mean_normal = np.mean([r['overall_normal'] for r in all_h1_ablation])
        mean_no_attn = np.mean([r['overall_no_attn'] for r in all_h1_ablation])
        mean_no_ffn = np.mean([r['overall_no_ffn'] for r in all_h1_ablation])
        std_no_attn = np.std([r['overall_no_attn'] for r in all_h1_ablation])
        std_no_ffn = np.std([r['overall_no_ffn'] for r in all_h1_ablation])
        print(f"  H1 Ablation: no_attn={mean_no_attn:.3f}+/-{std_no_attn:.3f} "
              f"no_ffn={mean_no_ffn:.3f}+/-{std_no_ffn:.3f}")

        # H1 probe means
        mean_attn_probe = np.mean([r['attn']['accuracy'] for r in all_h1_probes])
        mean_ffn_probe = np.mean([r['ffn']['accuracy'] for r in all_h1_probes])
        std_attn_probe = np.std([r['attn']['accuracy'] for r in all_h1_probes])
        std_ffn_probe = np.std([r['ffn']['accuracy'] for r in all_h1_probes])
        print(f"  H1 Probes: attn={mean_attn_probe:.3f}+/-{std_attn_probe:.3f} "
              f"ffn={mean_ffn_probe:.3f}+/-{std_ffn_probe:.3f}")

        if all_h3_routing:
            mean_MI = np.mean([r['MI'] for r in all_h3_routing])
            std_MI = np.std([r['MI'] for r in all_h3_routing])
            mean_nMI = np.mean([r['normalized_MI'] for r in all_h3_routing])
            print(f"  H3 Routing MI: {mean_MI:.3f}+/-{std_MI:.3f} "
                  f"(normalized: {mean_nMI:.3f})")
