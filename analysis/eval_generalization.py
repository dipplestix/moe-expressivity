"""
Generalization evaluation: test 3-digit-trained add-7 models on 4-digit inputs.

Tests whether learned carry circuits generalize to longer sequences.
Evaluates all 4 variants x 5 seeds, reports per-variant accuracy.
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


def num_to_reversed_digits(n, num_digits):
    digits = []
    for _ in range(num_digits):
        digits.append(n % 10)
        n //= 10
    return digits


def evaluate_on_digits(model, num_digits, num_samples=1000):
    """Evaluate model on num_digits-digit add-7 inputs."""
    model.eval()
    max_val = 10 ** num_digits - 1
    out_num_digits = num_digits + 1

    correct = 0
    total = num_samples

    with torch.no_grad():
        for _ in range(num_samples):
            n = torch.randint(0, max_val + 1, (1,)).item()
            expected = n + 7

            in_digits = num_to_reversed_digits(n, num_digits)
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

    return correct / total


def evaluate_by_carry_length(model, num_digits, num_samples_per_length=500):
    """Evaluate accuracy stratified by carry-chain length."""
    model.eval()
    max_val = 10 ** num_digits - 1
    out_num_digits = num_digits + 1

    # Group results by carry length
    results = {}

    with torch.no_grad():
        # Generate enough samples to get good coverage
        for _ in range(num_samples_per_length * (num_digits + 1)):
            n = torch.randint(0, max_val + 1, (1,)).item()
            expected = n + 7

            # Compute carry length
            carry = 0
            carry_len = 0
            temp = n
            for d in range(num_digits):
                digit = temp % 10
                temp //= 10
                s = digit + (7 if d == 0 else carry)
                if s >= 10:
                    carry = 1
                    carry_len = d + 1
                else:
                    carry = 0

            if carry_len not in results:
                results[carry_len] = {'correct': 0, 'total': 0}
            if results[carry_len]['total'] >= num_samples_per_length:
                continue

            in_digits = num_to_reversed_digits(n, num_digits)
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
            results[carry_len]['total'] += 1
            if pred_digits == expected_out:
                results[carry_len]['correct'] += 1

    return {k: v['correct'] / v['total'] if v['total'] > 0 else 0
            for k, v in sorted(results.items())}


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = ckpt['config']
    # Override max_seq_len to handle longer sequences
    model = OneLayerTransformer(max_seq_len=128, **cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def main():
    SEEDS = [42, 137, 256, 512, 1024]
    VARIANTS = ['ffn', 'glu', 'moe', 'moe_glu']
    TRAIN_DIGITS = 3
    TEST_DIGITS = 4
    NUM_SAMPLES = 1000

    print(f"{'Variant':<10} {'Seed':<6} {'3-digit (in-dist)':<20} {'4-digit (OOD)':<20}")
    print("=" * 60)

    summary = {v: {'in_dist': [], 'ood': []} for v in VARIANTS}

    for variant in VARIANTS:
        for seed in SEEDS:
            ckpt_path = f"checkpoints/add7_{variant}_nonorm_s{seed}/best_model.pt"
            if not os.path.exists(ckpt_path):
                print(f"{variant:<10} {seed:<6} MISSING")
                continue

            try:
                model = load_model(ckpt_path)
            except Exception as e:
                print(f"{variant:<10} {seed:<6} LOAD ERROR: {e}")
                continue

            # In-distribution eval
            acc_in = evaluate_on_digits(model, TRAIN_DIGITS, NUM_SAMPLES)
            # Out-of-distribution eval
            acc_ood = evaluate_on_digits(model, TEST_DIGITS, NUM_SAMPLES)

            summary[variant]['in_dist'].append(acc_in)
            summary[variant]['ood'].append(acc_ood)

            print(f"{variant:<10} {seed:<6} {acc_in:<20.1%} {acc_ood:<20.1%}")

        print()

    # Summary statistics
    print("\n" + "=" * 60)
    print(f"{'Variant':<10} {'3-digit (mean±std)':<22} {'4-digit (mean±std)':<22} {'Transfer %'}")
    print("=" * 60)

    for variant in VARIANTS:
        if not summary[variant]['in_dist']:
            continue
        in_arr = np.array(summary[variant]['in_dist'])
        ood_arr = np.array(summary[variant]['ood'])
        transfer = np.mean(ood_arr) / max(np.mean(in_arr), 1e-6) * 100

        print(f"{variant:<10} {np.mean(in_arr):.1%} ± {np.std(in_arr):.1%}      "
              f"{np.mean(ood_arr):.1%} ± {np.std(ood_arr):.1%}      "
              f"{transfer:.0f}%")

    # Carry-length breakdown on 4-digit (one seed per variant)
    print("\n\n4-digit accuracy by carry length (seed 42):")
    print("=" * 60)
    for variant in VARIANTS:
        ckpt_path = f"checkpoints/add7_{variant}_nonorm_s42/best_model.pt"
        if not os.path.exists(ckpt_path):
            continue
        try:
            model = load_model(ckpt_path)
        except Exception as e:
            print(f"{variant:<10} LOAD ERROR: {e}")
            continue
        by_carry = evaluate_by_carry_length(model, TEST_DIGITS, num_samples_per_length=500)
        carry_str = "  ".join(f"L={k}: {v:.1%}" for k, v in by_carry.items())
        print(f"{variant:<10} {carry_str}")


if __name__ == "__main__":
    main()
