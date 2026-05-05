"""
Training script for held-out generalization experiments on modular addition.

Experiments:
  1. --exclude_wrap : train on (a,b) where a+b < p (no modular reduction),
                      test on a+b >= p (requires mod operation)
  2. --exclude_operand_ge N : train on pairs where both a,b < N,
                              test on pairs with a >= N or b >= N

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


def build_pair_pools(p, exclude_wrap=False, exclude_operand_ge=None):
    """Split all p^2 pairs into train and held-out pools."""
    train_pool = []
    held_out_pool = []

    for a in range(p):
        for b in range(p):
            excluded = False

            if exclude_wrap:
                if a + b >= p:
                    excluded = True

            if exclude_operand_ge is not None:
                if a >= exclude_operand_ge or b >= exclude_operand_ge:
                    excluded = True

            if excluded:
                held_out_pool.append((a, b))
            else:
                train_pool.append((a, b))

    return train_pool, held_out_pool


def generate_batch_from_pool(pool, batch_size, p, device):
    """Generate a batch of modadd examples from a specific pair pool."""
    indices = torch.randint(0, len(pool), (batch_size,))
    pairs = [pool[i] for i in indices]
    a = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
    b = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
    targets = (a + b) % p
    eq = torch.full_like(a, p)  # = token
    inputs = torch.stack([a, b, eq], dim=1)  # (B, 3)
    return inputs, targets


def evaluate_on_pairs(model, pairs, p, device):
    """Evaluate exact-match accuracy on specific pairs."""
    model.eval()
    if len(pairs) == 0:
        return 0.0, 0

    # Cap at 5000 for speed
    if len(pairs) > 5000:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(pairs), 5000, replace=False)
        pairs = [pairs[i] for i in idx]

    a = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
    b = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
    targets = (a + b) % p
    eq = torch.full_like(a, p)
    inputs = torch.stack([a, b, eq], dim=1)

    with torch.no_grad():
        logits = model(inputs)
        preds = logits[:, 2, :].argmax(dim=-1)
        correct = (preds == targets).sum().item()

    model.train()
    return correct / len(pairs), len(pairs)


def evaluate_all_subsets(model, p, device, train_pool, held_out_pool):
    """Evaluate on train pool, held-out pool, and per-subset breakdowns."""
    results = {}

    acc_train, n_train = evaluate_on_pairs(model, train_pool, p, device)
    acc_held, n_held = evaluate_on_pairs(model, held_out_pool, p, device)
    results['train_pool'] = {'acc': acc_train, 'n': n_train}
    results['held_out_pool'] = {'acc': acc_held, 'n': n_held}

    # Wrap vs no-wrap
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    wrap = [(a, b) for a, b in all_pairs if a + b >= p]
    no_wrap = [(a, b) for a, b in all_pairs if a + b < p]
    acc_w, n_w = evaluate_on_pairs(model, wrap, p, device)
    acc_nw, n_nw = evaluate_on_pairs(model, no_wrap, p, device)
    results['wrap'] = {'acc': acc_w, 'n': n_w}
    results['no_wrap'] = {'acc': acc_nw, 'n': n_nw}

    # Overall
    acc_all, n_all = evaluate_on_pairs(model, all_pairs, p, device)
    results['all'] = {'acc': acc_all, 'n': n_all}

    return results


def main():
    parser = argparse.ArgumentParser(description="Train modadd with held-out generalization")

    # Task
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--seed', type=int, default=42)

    # Exclusion criteria
    parser.add_argument('--exclude_wrap', action='store_true',
                        help='Exclude pairs where a+b >= p (modular wrap-around)')
    parser.add_argument('--exclude_operand_ge', type=int, default=None,
                        help='Exclude pairs where a >= N or b >= N')

    # Model
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ffn_type', type=str, default='ffn',
                        choices=['ffn', 'glu', 'moe', 'moe_glu'])
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--moe_balance_coeff', type=float, default=0.01)
    parser.add_argument('--use_norm', action='store_true', default=False)
    parser.add_argument('--intermediate_dim', type=int, default=512)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--dropout', type=float, default=0.0)

    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=40000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--patience', type=int, default=200,
                        help='Early stop after this many evals without train improvement')

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--no_wandb', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build pools
    train_pool, held_out_pool = build_pair_pools(
        args.p,
        exclude_wrap=args.exclude_wrap,
        exclude_operand_ge=args.exclude_operand_ge,
    )

    if args.exclude_wrap:
        exp_name = "exclude_wrap"
    elif args.exclude_operand_ge is not None:
        exp_name = f"exclude_operand_ge{args.exclude_operand_ge}"
    else:
        exp_name = "baseline"

    print(f"Experiment: {exp_name}")
    print(f"Train pool: {len(train_pool)} pairs")
    print(f"Held-out pool: {len(held_out_pool)} pairs")

    if len(train_pool) == 0:
        print("ERROR: empty train pool!")
        return

    # Model
    model = OneLayerTransformer(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ffn_type=args.ffn_type,
        vocab_size=args.p + 1,
        max_seq_len=3,
        use_norm=args.use_norm,
        is_causal=False,
        tie_embeddings=False,
        activation=args.activation,
        dropout=args.dropout,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    best_train_acc = 0.0
    patience_counter = 0

    # Full-batch training on the train pool
    train_a = torch.tensor([p[0] for p in train_pool], dtype=torch.long, device=device)
    train_b = torch.tensor([p[1] for p in train_pool], dtype=torch.long, device=device)
    train_targets = (train_a + train_b) % args.p
    train_eq = torch.full_like(train_a, args.p)
    train_inputs = torch.stack([train_a, train_b, train_eq], dim=1)

    model.train()
    for epoch in range(1, args.epochs + 1):
        # Shuffle
        perm = torch.randperm(len(train_inputs), device=device)
        inputs_shuffled = train_inputs[perm]
        targets_shuffled = train_targets[perm]

        logits = model(inputs_shuffled)
        loss = F.cross_entropy(logits[:, 2, :], targets_shuffled)

        if args.ffn_type in ('moe', 'moe_glu'):
            loss = loss + args.moe_balance_coeff * model._aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_inputs)
                train_preds = train_logits[:, 2, :].argmax(dim=-1)
                train_acc = (train_preds == train_targets).float().mean().item()

            acc_held, _ = evaluate_on_pairs(model, held_out_pool, args.p, device)

            print(f"Epoch {epoch}: loss={loss.item():.4f} "
                  f"train_acc={train_acc:.1%} held_out_acc={acc_held:.1%}")

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'test_acc': train_acc,
                    'config': {
                        'model_dim': args.model_dim,
                        'num_heads': args.num_heads,
                        'ffn_type': args.ffn_type,
                        'vocab_size': args.p + 1,
                        'max_seq_len': 3,
                        'use_norm': args.use_norm,
                        'is_causal': False,
                        'tie_embeddings': False,
                        'activation': args.activation,
                        'dropout': args.dropout,
                        'intermediate_dim': args.intermediate_dim,
                        'num_experts': args.num_experts,
                        'top_k': args.top_k,
                    },
                    'args': vars(args),
                    'exp_name': exp_name,
                }, ckpt_dir / 'modadd_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            model.train()

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    results = evaluate_all_subsets(model, args.p, device, train_pool, held_out_pool)

    for name, data in results.items():
        print(f"  {name:<20} acc={data['acc']:.1%}  (n={data['n']})")

    results_path = ckpt_dir / 'generalization_results.json'
    json_results = {
        'exp_name': exp_name,
        'variant': args.ffn_type,
        'seed': args.seed,
        'p': args.p,
        'train_pool_size': len(train_pool),
        'held_out_pool_size': len(held_out_pool),
        'results': {k: {'acc': v['acc'], 'n': v['n']} for k, v in results.items()},
    }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
