"""
Training script for held-out generalization experiments on add-7.

Two experiments:
  1. --exclude_carry_ge N : train excluding examples with carry length >= N,
                            test on both in-distribution and held-out carry lengths
  2. --exclude_ending_9   : train excluding numbers where ones digit >= 3
                            (ones digit >= 3 means +7 causes carry),
                            test on both subsets

Saves detailed per-subset eval results in a JSON alongside the checkpoint.
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, 'model')
from model import OneLayerTransformer

try:
    import wandb
except ImportError:
    wandb = None

PAD_TOKEN = 10
EOS_TOKEN = 11
VOCAB_SIZE = 12


def num_to_reversed_digits(n, num_digits):
    digits = []
    for _ in range(num_digits):
        digits.append(n % 10)
        n //= 10
    return digits


def compute_carry_length(n, num_digits):
    """Compute carry-chain length for n+7."""
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
    return carry_len


def build_number_pools(num_digits, exclude_carry_ge=None, exclude_ending_9=False,
                       exclude_tens_9=False):
    """Split all numbers into train and held-out pools based on exclusion criteria."""
    max_val = 10 ** num_digits - 1
    train_pool = []
    held_out_pool = []

    for n in range(max_val + 1):
        excluded = False

        if exclude_carry_ge is not None:
            cl = compute_carry_length(n, num_digits)
            if cl >= exclude_carry_ge:
                excluded = True

        if exclude_ending_9:
            # Exclude numbers where ones digit >= 3 (adding 7 causes overflow/carry)
            if n % 10 >= 3:
                excluded = True

        if exclude_tens_9:
            # Exclude numbers where tens digit = 9 (carry can't propagate through tens)
            if (n // 10) % 10 == 9:
                excluded = True

        if excluded:
            held_out_pool.append(n)
        else:
            train_pool.append(n)

    return train_pool, held_out_pool


def generate_batch_from_pool(pool, batch_size, num_digits, device):
    """Generate a batch of add-7 examples from a specific number pool."""
    indices = torch.randint(0, len(pool), (batch_size,))
    inputs = torch.tensor([pool[i] for i in indices])
    outputs = inputs + 7

    out_num_digits = num_digits + 1

    sequences = []
    for i in range(batch_size):
        in_digits = num_to_reversed_digits(inputs[i].item(), num_digits)
        out_digits = num_to_reversed_digits(outputs[i].item(), out_num_digits)
        seq = in_digits + [EOS_TOKEN] + out_digits + [EOS_TOKEN]
        sequences.append(seq)

    sequences = torch.tensor(sequences, dtype=torch.long, device=device)
    return sequences, num_digits


def compute_loss(model, sequences, input_len, moe_balance_coeff=0.0, ffn_type='ffn'):
    x = sequences[:, :-1]
    targets = sequences[:, 1:]

    logits = model(x)

    mask = torch.zeros_like(targets, dtype=torch.float)
    mask[:, input_len:] = 1.0

    B, T = targets.shape
    logits_flat = logits.reshape(B * T, -1)
    targets_flat = targets.reshape(B * T)
    mask_flat = mask.reshape(B * T)

    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    loss = (loss_per_token * mask_flat).sum() / mask_flat.sum()

    if ffn_type in ('moe', 'moe_glu'):
        loss = loss + moe_balance_coeff * model._aux_loss

    return loss


def evaluate_on_numbers(model, numbers, num_digits, device):
    """Evaluate exact-match accuracy on a specific set of numbers."""
    model.eval()
    out_num_digits = num_digits + 1
    correct = 0
    total = len(numbers)

    if total == 0:
        return 0.0, 0

    # Cap at 1000 for speed
    if total > 1000:
        rng = np.random.RandomState(0)
        numbers = rng.choice(numbers, 1000, replace=False).tolist()
        total = 1000

    with torch.no_grad():
        for n in numbers:
            expected = n + 7
            in_digits = num_to_reversed_digits(n, num_digits)
            expected_out = num_to_reversed_digits(expected, out_num_digits)

            seq = torch.tensor([in_digits + [EOS_TOKEN]], dtype=torch.long, device=device)

            generated = []
            for _ in range(out_num_digits + 1):
                logits = model(seq)
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)
                if next_token == EOS_TOKEN:
                    break
                seq = torch.cat([seq, torch.tensor([[next_token]], device=device)], dim=1)

            pred_digits = [t for t in generated if t != EOS_TOKEN]
            if pred_digits == expected_out:
                correct += 1

    model.train()
    return correct / total, total


def evaluate_all_subsets(model, num_digits, device, train_pool, held_out_pool):
    """Evaluate on train pool, held-out pool, and per-carry-length."""
    results = {}

    acc_train, n_train = evaluate_on_numbers(model, train_pool, num_digits, device)
    acc_held, n_held = evaluate_on_numbers(model, held_out_pool, num_digits, device)
    results['train_pool'] = {'acc': acc_train, 'n': n_train}
    results['held_out_pool'] = {'acc': acc_held, 'n': n_held}

    # Per carry length
    max_val = 10 ** num_digits - 1
    by_carry = {}
    for n in range(max_val + 1):
        cl = compute_carry_length(n, num_digits)
        by_carry.setdefault(cl, []).append(n)

    for cl in sorted(by_carry.keys()):
        acc, n = evaluate_on_numbers(model, by_carry[cl], num_digits, device)
        results[f'L={cl}'] = {'acc': acc, 'n': n}

    # Ends in 9 vs not
    ends_9 = [n for n in range(max_val + 1) if n % 10 == 9]
    not_ends_9 = [n for n in range(max_val + 1) if n % 10 != 9]
    acc_9, n_9 = evaluate_on_numbers(model, ends_9, num_digits, device)
    acc_not9, n_not9 = evaluate_on_numbers(model, not_ends_9, num_digits, device)
    results['ends_in_9'] = {'acc': acc_9, 'n': n_9}
    results['not_ends_in_9'] = {'acc': acc_not9, 'n': n_not9}

    # Ones >= 3 (carry-triggering)
    ones_ge3 = [n for n in range(max_val + 1) if n % 10 >= 3]
    ones_lt3 = [n for n in range(max_val + 1) if n % 10 < 3]
    acc_ge3, n_ge3 = evaluate_on_numbers(model, ones_ge3, num_digits, device)
    acc_lt3, n_lt3 = evaluate_on_numbers(model, ones_lt3, num_digits, device)
    results['ones_ge3'] = {'acc': acc_ge3, 'n': n_ge3}
    results['ones_lt3'] = {'acc': acc_lt3, 'n': n_lt3}

    return results


def main():
    parser = argparse.ArgumentParser(description="Train add-7 with held-out generalization")

    # Task
    parser.add_argument('--num_digits', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)

    # Exclusion criteria (pick one)
    parser.add_argument('--exclude_carry_ge', type=int, default=None,
                        help='Exclude training examples with carry length >= N')
    parser.add_argument('--exclude_ending_9', action='store_true',
                        help='Exclude training examples where ones digit >= 3')
    parser.add_argument('--exclude_tens_9', action='store_true',
                        help='Exclude training examples where tens digit = 9')

    # Model
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ffn_type', type=str, default='ffn',
                        choices=['ffn', 'glu', 'moe', 'moe_glu'])
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--moe_balance_coeff', type=float, default=0.01)
    parser.add_argument('--activation', type=str, default='gelu',
                        choices=['gelu', 'silu', 'relu'])
    parser.add_argument('--use_norm', dest='use_norm', action='store_true', default=False)
    parser.add_argument('--no_use_norm', dest='use_norm', action='store_false')

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--patience', type=int, default=50)

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project', type=str, default='add7-generalization')
    parser.add_argument('--no_wandb', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build train/held-out pools
    train_pool, held_out_pool = build_number_pools(
        args.num_digits,
        exclude_carry_ge=args.exclude_carry_ge,
        exclude_ending_9=args.exclude_ending_9,
        exclude_tens_9=args.exclude_tens_9,
    )

    # Describe the experiment
    if args.exclude_carry_ge is not None:
        exp_name = f"exclude_carry_ge{args.exclude_carry_ge}"
    elif args.exclude_ending_9:
        exp_name = "exclude_ones_ge3"
    elif args.exclude_tens_9:
        exp_name = "exclude_tens_9"
    else:
        exp_name = "baseline"

    print(f"Experiment: {exp_name}")
    print(f"Train pool: {len(train_pool)} numbers")
    print(f"Held-out pool: {len(held_out_pool)} numbers")

    if len(train_pool) == 0:
        print("ERROR: empty train pool!")
        return

    # Model
    model = OneLayerTransformer(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ffn_type=args.ffn_type,
        vocab_size=VOCAB_SIZE,
        use_norm=args.use_norm,
        num_experts=args.num_experts,
        top_k=args.top_k,
        activation=args.activation,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, config={**vars(args), 'exp_name': exp_name})

    best_acc = 0.0
    patience_counter = 0

    model.train()
    for step in range(1, args.steps + 1):
        sequences, input_len = generate_batch_from_pool(
            train_pool, args.batch_size, args.num_digits, device
        )

        optimizer.zero_grad()
        loss = compute_loss(model, sequences, input_len,
                           args.moe_balance_coeff, args.ffn_type)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % args.eval_interval == 0:
            # Quick eval on train pool only for logging
            acc_train, _ = evaluate_on_numbers(
                model, train_pool[:500], args.num_digits, device
            )
            acc_held, _ = evaluate_on_numbers(
                model, held_out_pool[:500] if held_out_pool else [],
                args.num_digits, device
            )

            print(f"Step {step}: loss={loss.item():.4f} "
                  f"train_acc={acc_train:.1%} held_out_acc={acc_held:.1%}")

            log_dict = {
                'step': step, 'loss': loss.item(),
                'train_acc': acc_train, 'held_out_acc': acc_held,
            }
            if use_wandb:
                wandb.log(log_dict)

            if acc_train > best_acc:
                best_acc = acc_train
                patience_counter = 0

                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': acc_train,
                    'held_out_acc': acc_held,
                    'config': {
                        'model_dim': args.model_dim,
                        'num_heads': args.num_heads,
                        'ffn_type': args.ffn_type,
                        'vocab_size': VOCAB_SIZE,
                        'use_norm': args.use_norm,
                        'num_experts': args.num_experts,
                        'top_k': args.top_k,
                    },
                    'args': vars(args),
                    'exp_name': exp_name,
                }, ckpt_dir / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at step {step}")
                    break

    # Final comprehensive evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    results = evaluate_all_subsets(
        model, args.num_digits, device, train_pool, held_out_pool
    )

    for name, data in results.items():
        print(f"  {name:<20} acc={data['acc']:.1%}  (n={data['n']})")

    # Save results JSON
    results_path = ckpt_dir / 'generalization_results.json'
    json_results = {
        'exp_name': exp_name,
        'variant': args.ffn_type,
        'seed': args.seed,
        'num_digits': args.num_digits,
        'train_pool_size': len(train_pool),
        'held_out_pool_size': len(held_out_pool),
        'results': {k: {'acc': v['acc'], 'n': v['n']} for k, v in results.items()},
    }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if use_wandb:
        wandb.log({f'final/{k}': v['acc'] for k, v in results.items()})
        wandb.finish()


if __name__ == '__main__':
    main()
