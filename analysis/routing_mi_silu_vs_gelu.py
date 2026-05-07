"""
Compute routing MI and per-expert operation distributions for MoE-GLU models.
Compares SiLU-trained vs GELU-trained checkpoints on add-7 (3-digit).
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, "<PATH_TO_REPO>/model")
from model import OneLayerTransformer

# Constants
PAD_TOKEN = 10
EOS_TOKEN = 11
VOCAB_SIZE = 12
NUM_DIGITS = 3
SEEDS = [42, 137, 256, 512, 1024]


def num_to_reversed_digits(n, num_digits):
    digits = []
    for _ in range(num_digits):
        digits.append(n % 10)
        n //= 10
    return digits


def generate_all_inputs(num_digits, device):
    """Generate all inputs 0..999 for 3-digit add-7."""
    max_val = 10 ** num_digits - 1
    out_num_digits = num_digits + 1

    sequences = []
    for n in range(max_val + 1):
        result = n + 7
        in_digits = num_to_reversed_digits(n, num_digits)
        out_digits = num_to_reversed_digits(result, out_num_digits)
        seq = in_digits + [EOS_TOKEN] + out_digits + [EOS_TOKEN]
        sequences.append(seq)

    return torch.tensor(sequences, dtype=torch.long, device=device)


def get_operation_labels(num_digits):
    """For each input 0..999, get per-output-position operation type.

    Output positions (after EOS):
      pos 0 (ones): always +7
      pos 1 (tens): +1 if carry from ones, else +0
      pos 2 (hundreds): +1 if carry propagates, else +0
      pos 3 (thousands): +1 if carry propagates, else +0

    Returns: array of shape (1000, out_num_digits) with values in {'+7', '+1', '+0'}
    """
    max_val = 10 ** num_digits - 1
    out_num_digits = num_digits + 1
    labels = []

    for n in range(max_val + 1):
        ops = []
        # Simulate digit-by-digit addition
        carry = 0
        in_digits = num_to_reversed_digits(n, num_digits)
        for pos in range(out_num_digits):
            d = in_digits[pos] if pos < num_digits else 0
            if pos == 0:
                # ones place: add 7
                ops.append("+7")
                total = d + 7
                carry = total // 10
            else:
                if carry == 1:
                    ops.append("+1")
                    total = d + carry
                    carry = total // 10
                else:
                    ops.append("+0")
                    carry = 0
        labels.append(ops)

    return labels


def load_model(ckpt_path, activation="silu"):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model = OneLayerTransformer(
        model_dim=config["model_dim"],
        num_heads=config["num_heads"],
        ffn_type=config["ffn_type"],
        vocab_size=config["vocab_size"],
        use_norm=config.get("use_norm", True),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 1),
        intermediate_dim=config.get("intermediate_dim", None),
        activation=activation,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("accuracy", None)


def get_expert_assignments(model, sequences):
    """Run model and capture expert assignments at each position.

    Returns: expert_ids of shape (B, T) -- top-1 expert index for each token.
    """
    B, T_full = sequences.shape
    x = sequences[:, :-1]  # input tokens (drop last)
    B, T = x.shape

    # Forward pass up to the router
    with torch.no_grad():
        pos = torch.arange(T, device=x.device)
        h = model.vocab(x) + model.pos_embed(pos)
        h = h + model.atn(model.atn_norm(h))

        # Get FFN input (after norm)
        ffn_input = model.ffn_norm(h)

        # Get router logits from MoEGLU
        moe = model.ffn
        x_flat = ffn_input.view(B * T, -1)
        router_logits = moe.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        _, topk_indices = torch.topk(router_probs, moe.top_k, dim=-1)

        # top-1 expert assignment
        expert_ids = topk_indices[:, 0].view(B, T)

    return expert_ids


def compute_routing_mi(expert_ids, operation_labels, num_digits):
    """Compute NMI between expert assignment and operation type at output positions.

    Output positions in the x (input to model, sequences[:,:-1]) correspond to:
      - Position num_digits is EOS (predicts first output digit)
      - Position num_digits+1 predicts second output digit
      - etc.

    So output positions in expert_ids are [num_digits, num_digits+1, ..., num_digits+out_num_digits-1]
    which predict output digits [0, 1, ..., out_num_digits-1].
    """
    out_num_digits = num_digits + 1
    all_experts = []
    all_ops = []

    for out_pos in range(out_num_digits):
        # Position in the sequence (input tokens) that predicts this output digit
        seq_pos = num_digits + out_pos  # EOS is at num_digits, then output digits follow
        experts_at_pos = expert_ids[:, seq_pos].numpy()
        ops_at_pos = [operation_labels[i][out_pos] for i in range(len(operation_labels))]

        all_experts.extend(experts_at_pos.tolist())
        all_ops.extend(ops_at_pos)

    nmi = normalized_mutual_info_score(all_ops, all_experts)
    return nmi


def compute_per_expert_distribution(expert_ids, operation_labels, num_digits, num_experts=4):
    """Compute what fraction of each operation type goes to each expert."""
    out_num_digits = num_digits + 1
    # Count: expert_counts[op][expert] = count
    from collections import Counter
    op_expert_counts = {"+7": Counter(), "+1": Counter(), "+0": Counter()}

    for out_pos in range(out_num_digits):
        seq_pos = num_digits + out_pos
        for i in range(expert_ids.shape[0]):
            e = expert_ids[i, seq_pos].item()
            op = operation_labels[i][out_pos]
            op_expert_counts[op][e] += 1

    # Normalize to fractions
    result = {}
    for op in ["+7", "+1", "+0"]:
        total = sum(op_expert_counts[op].values())
        if total > 0:
            result[op] = {e: op_expert_counts[op][e] / total for e in range(num_experts)}
        else:
            result[op] = {e: 0.0 for e in range(num_experts)}

    return result


def run_analysis(label, ckpt_template, activation, seeds):
    """Run full analysis for a set of checkpoints."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    sequences = generate_all_inputs(NUM_DIGITS, "cpu")
    operation_labels = get_operation_labels(NUM_DIGITS)
    nmis = []

    for seed in seeds:
        ckpt_path = ckpt_template.format(seed=seed)
        if not Path(ckpt_path).exists():
            print(f"  [SKIP] {ckpt_path} not found")
            continue

        model, acc = load_model(ckpt_path, activation=activation)
        expert_ids = get_expert_assignments(model, sequences)
        nmi = compute_routing_mi(expert_ids, operation_labels, NUM_DIGITS)
        dist = compute_per_expert_distribution(expert_ids, operation_labels, NUM_DIGITS)
        nmis.append(nmi)

        print(f"\n  Seed {seed} (acc={acc}):")
        print(f"    NMI = {nmi:.4f}")
        print(f"    Per-expert operation distribution:")
        for op in ["+7", "+1", "+0"]:
            fracs = [f"E{e}={dist[op][e]:.2%}" for e in range(4)]
            print(f"      {op}: {', '.join(fracs)}")

    if nmis:
        nmis = np.array(nmis)
        print(f"\n  Summary: NMI = {nmis.mean():.4f} +/- {nmis.std():.4f} (n={len(nmis)})")
        print(f"  Individual NMIs: {[f'{x:.4f}' for x in nmis]}")

    return nmis


def main():
    base = Path("<PATH_TO_REPO>/checkpoints")

    silu_nmis = run_analysis(
        "SiLU MoE-GLU (nonorm)",
        str(base / "add7_moe_glu_silu_nonorm_s{seed}/best_model.pt"),
        activation="silu",
        seeds=SEEDS,
    )

    gelu_nmis = run_analysis(
        "GELU MoE-GLU (nonorm) -- baseline",
        str(base / "add7_moe_glu_nonorm_s{seed}/best_model.pt"),
        activation="gelu",
        seeds=SEEDS,
    )

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON: SiLU vs GELU")
    print(f"{'='*60}")
    if len(silu_nmis) > 0 and len(gelu_nmis) > 0:
        print(f"  SiLU NMI: {np.mean(silu_nmis):.4f} +/- {np.std(silu_nmis):.4f}")
        print(f"  GELU NMI: {np.mean(gelu_nmis):.4f} +/- {np.std(gelu_nmis):.4f}")
        diff = np.mean(silu_nmis) - np.mean(gelu_nmis)
        print(f"  Difference (SiLU - GELU): {diff:+.4f}")
    else:
        print("  Insufficient data for comparison.")


if __name__ == "__main__":
    main()
