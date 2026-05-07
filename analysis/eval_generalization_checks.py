"""
Generalization checks from original experiment plan.

Tests existing 3-digit add-7 models on filtered subsets:
1. L>=3 carry chains only (do models generalize to long carries?)
2. Numbers ending in 9 only (carry-triggering inputs)
3. Numbers NOT ending in 9 (no-carry baseline)
4. Per-carry-length breakdown (L=0,1,2,3)

All use existing checkpoints — no retraining needed.
"""

import sys
import os
os.chdir("<PATH_TO_REPO>")
sys.path.insert(0, ".")
sys.path.insert(0, "model")

import torch
import numpy as np
from model import OneLayerTransformer

PAD_TOKEN = 10
EOS_TOKEN = 11
VOCAB_SIZE = 12
DEVICE = "cpu"
NUM_DIGITS = 3


def num_to_reversed_digits(n, num_digits):
    digits = []
    for _ in range(num_digits):
        digits.append(n % 10)
        n //= 10
    return digits


def compute_carry_length(n):
    """Compute carry-chain length for n+7."""
    carry = 0
    carry_len = 0
    temp = n
    for d in range(NUM_DIGITS):
        digit = temp % 10
        temp //= 10
        s = digit + (7 if d == 0 else carry)
        if s >= 10:
            carry = 1
            carry_len = d + 1
        else:
            carry = 0
    return carry_len


def evaluate_on_subset(model, numbers):
    """Evaluate model on a specific set of numbers."""
    model.eval()
    out_num_digits = NUM_DIGITS + 1
    correct = 0
    total = len(numbers)

    with torch.no_grad():
        for n in numbers:
            expected = n + 7
            in_digits = num_to_reversed_digits(n, NUM_DIGITS)
            expected_out = num_to_reversed_digits(expected, out_num_digits)

            seq = torch.tensor([in_digits + [EOS_TOKEN]], dtype=torch.long, device=DEVICE)

            generated = []
            for _ in range(out_num_digits + 1):
                logits = model(seq)
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)
                if next_token == EOS_TOKEN:
                    break
                seq = torch.cat([seq, torch.tensor([[next_token]], device=DEVICE)], dim=1)

            pred_digits = [t for t in generated if t != EOS_TOKEN]
            if pred_digits == expected_out:
                correct += 1

    return correct / total if total > 0 else 0.0


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = ckpt['config']
    model = OneLayerTransformer(max_seq_len=128, **cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def main():
    SEEDS = [42, 137, 256, 512, 1024]
    VARIANTS = ['ffn', 'glu', 'moe', 'moe_glu']

    # Generate all 3-digit numbers and categorize
    all_numbers = list(range(0, 10**NUM_DIGITS))

    by_carry = {}
    ends_in_9 = []
    not_ends_in_9 = []

    for n in all_numbers:
        cl = compute_carry_length(n)
        by_carry.setdefault(cl, []).append(n)
        if n % 10 >= 3:  # ones digit >= 3 means +7 overflows
            ends_in_9_flag = False
            # Check if ones digit is such that carry propagates
        # Simpler: just check if number ends in 9
        if (n % 10) == 9:
            ends_in_9.append(n)
        else:
            not_ends_in_9.append(n)

    print("=== Data Distribution ===")
    for cl in sorted(by_carry.keys()):
        print(f"  L={cl}: {len(by_carry[cl])} numbers ({100*len(by_carry[cl])/len(all_numbers):.1f}%)")
    print(f"  Ends in 9: {len(ends_in_9)} ({100*len(ends_in_9)/len(all_numbers):.1f}%)")
    print(f"  Not ends in 9: {len(not_ends_in_9)} ({100*len(not_ends_in_9)/len(all_numbers):.1f}%)")
    print()

    # Also: numbers where tens digit is 9 (carry can propagate further)
    tens_is_9 = [n for n in all_numbers if (n // 10) % 10 == 9]
    ends_ge3_and_tens9 = [n for n in all_numbers if n % 10 >= 3 and (n // 10) % 10 == 9]
    print(f"  Tens digit is 9: {len(tens_is_9)}")
    print(f"  Ones>=3 AND tens=9 (L>=2): {len(ends_ge3_and_tens9)}")
    print()

    # Test subsets
    subsets = {
        'All': all_numbers,
        'L=0 (no carry)': by_carry.get(0, []),
        'L=1 (one carry)': by_carry.get(1, []),
        'L=2 (two carries)': by_carry.get(2, []),
        'L=3 (three carries)': by_carry.get(3, []),
        'Ends in 9': ends_in_9,
        'Not ends in 9': not_ends_in_9,
        'Ones>=3 & tens=9': ends_ge3_and_tens9,
    }

    # Results storage
    results = {v: {s: [] for s in subsets} for v in VARIANTS}

    for variant in VARIANTS:
        print(f"\n{'='*70}")
        print(f"Variant: {variant}")
        print(f"{'='*70}")

        for seed in SEEDS:
            ckpt_path = f"checkpoints/add7_{variant}_nonorm_s{seed}/best_model.pt"
            if not os.path.exists(ckpt_path):
                print(f"  seed={seed}: MISSING")
                continue

            try:
                model = load_model(ckpt_path)
            except Exception as e:
                print(f"  seed={seed}: LOAD ERROR: {e}")
                continue

            print(f"  seed={seed}:", end="")
            for name, nums in subsets.items():
                if len(nums) == 0:
                    print(f"  {name}=N/A", end="")
                    continue
                # Sample if subset is large
                if len(nums) > 1000:
                    sample = np.random.RandomState(seed).choice(nums, 1000, replace=False).tolist()
                else:
                    sample = nums
                acc = evaluate_on_subset(model, sample)
                results[variant][name].append(acc)
                print(f"  {name}={acc:.1%}", end="")
            print()

    # Summary table
    print(f"\n\n{'='*90}")
    print("SUMMARY (mean +/- std across seeds)")
    print(f"{'='*90}")
    header = f"{'Variant':<10}"
    for name in subsets:
        header += f" {name:<18}"
    print(header)
    print("-" * 90)

    for variant in VARIANTS:
        row = f"{variant:<10}"
        for name in subsets:
            vals = results[variant][name]
            if len(vals) == 0:
                row += f" {'N/A':<18}"
            else:
                m, s = np.mean(vals), np.std(vals)
                row += f" {m:.1%} +/- {s:.1%}   "
        print(row)


if __name__ == "__main__":
    main()
