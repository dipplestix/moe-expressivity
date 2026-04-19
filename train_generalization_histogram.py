"""
Training script for held-out generalization experiments on histogram counting.

Experiments:
  1. --exclude_high_count N : train excluding sequences where any token count >= N,
                              test on sequences with high counts
  2. --exclude_tokens LIST : train excluding specific token values from appearing,
                             test on sequences containing those tokens

Saves detailed per-subset eval results in a JSON alongside the checkpoint.
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
sys.path.insert(0, 'model')
from model import OneLayerTransformer
from data.histogram import HistogramDataset

try:
    import wandb
except ImportError:
    wandb = None


def filter_dataset(inputs, targets, exclude_high_count=None, exclude_tokens=None):
    """Filter a dataset based on exclusion criteria. Returns train and held-out masks."""
    n = inputs.shape[0]
    excluded = torch.zeros(n, dtype=torch.bool)

    if exclude_high_count is not None:
        # targets are counts-1 (class indices), so count >= N means target >= N-1
        max_counts = targets.max(dim=1).values  # max count class per sequence
        excluded |= (max_counts >= exclude_high_count - 1)

    if exclude_tokens is not None:
        for t in exclude_tokens:
            has_token = (inputs == t).any(dim=1)
            excluded |= has_token

    return ~excluded, excluded


def evaluate(model, inputs, targets):
    """Evaluate accuracy on a dataset."""
    model.eval()
    if inputs.shape[0] == 0:
        return 0.0, 0

    with torch.no_grad():
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float().mean().item()

    return correct, inputs.shape[0]


def main():
    parser = argparse.ArgumentParser(description="Train histogram with held-out generalization")

    # Task
    parser.add_argument('--T', type=int, default=32, help='Alphabet size')
    parser.add_argument('--L', type=int, default=10, help='Sequence length')
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_test', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--refresh_interval', type=int, default=50)

    # Exclusion criteria
    parser.add_argument('--exclude_high_count', type=int, default=None,
                        help='Exclude sequences where any token count >= N')
    parser.add_argument('--exclude_tokens', type=int, nargs='+', default=None,
                        help='Exclude sequences containing these token values')

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
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--patience', type=int, default=100)

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--no_wandb', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.exclude_high_count is not None:
        exp_name = f"exclude_highcount{args.exclude_high_count}"
    elif args.exclude_tokens is not None:
        exp_name = f"exclude_tokens_{'_'.join(map(str, args.exclude_tokens))}"
    else:
        exp_name = "baseline"

    print(f"Experiment: {exp_name}")

    # Model
    model = OneLayerTransformer(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ffn_type=args.ffn_type,
        vocab_size=args.T,
        max_seq_len=args.L,
        use_norm=args.use_norm,
        is_causal=False,
        tie_embeddings=False,
        activation=args.activation,
        dropout=args.dropout,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_classes=args.L,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    best_train_acc = 0.0
    patience_counter = 0

    # Generate initial dataset and filter
    rs = np.random.RandomState(args.seed)
    dataset = HistogramDataset(T=args.T, L=args.L, n_train=args.n_train,
                               n_test=args.n_test, seed=args.seed, device=device)

    # We always evaluate on the FULL unfiltered test set
    full_test_inputs = dataset.test_inputs.clone()
    full_test_targets = dataset.test_targets.clone()

    # Also create filtered subsets for detailed eval
    test_train_mask, test_held_mask = filter_dataset(
        full_test_inputs, full_test_targets,
        exclude_high_count=args.exclude_high_count,
        exclude_tokens=args.exclude_tokens,
    )

    model.train()
    for epoch in range(1, args.epochs + 1):
        # Refresh data periodically
        if epoch % args.refresh_interval == 0:
            dataset = HistogramDataset(T=args.T, L=args.L, n_train=args.n_train,
                                       n_test=args.n_test, seed=args.seed + epoch,
                                       device=device)

        # Filter training data
        train_mask, _ = filter_dataset(
            dataset.train_inputs, dataset.train_targets,
            exclude_high_count=args.exclude_high_count,
            exclude_tokens=args.exclude_tokens,
        )
        train_inputs = dataset.train_inputs[train_mask]
        train_targets = dataset.train_targets[train_mask]

        if train_inputs.shape[0] == 0:
            continue

        logits = model(train_inputs)
        B, T_seq, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T_seq, C), train_targets.reshape(B * T_seq))

        if args.ffn_type in ('moe', 'moe_glu'):
            loss = loss + args.moe_balance_coeff * model._aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            model.eval()
            with torch.no_grad():
                # Train acc (on filtered)
                t_logits = model(train_inputs)
                t_preds = t_logits.argmax(dim=-1)
                train_acc = (t_preds == train_targets).float().mean().item()

                # Full test acc
                full_logits = model(full_test_inputs)
                full_preds = full_logits.argmax(dim=-1)
                full_acc = (full_preds == full_test_targets).float().mean().item()

                # Held-out subset acc
                if test_held_mask.any():
                    held_preds = full_preds[test_held_mask]
                    held_targets = full_test_targets[test_held_mask]
                    held_acc = (held_preds == held_targets).float().mean().item()
                else:
                    held_acc = 0.0

            print(f"Epoch {epoch}: loss={loss.item():.4f} "
                  f"train_acc={train_acc:.1%} full_test={full_acc:.1%} held_out={held_acc:.1%}")

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
                        'vocab_size': args.T,
                        'max_seq_len': args.L,
                        'use_norm': args.use_norm,
                        'is_causal': False,
                        'tie_embeddings': False,
                        'activation': args.activation,
                        'dropout': args.dropout,
                        'intermediate_dim': args.intermediate_dim,
                        'num_experts': args.num_experts,
                        'top_k': args.top_k,
                        'num_classes': args.L,
                    },
                    'args': vars(args),
                    'exp_name': exp_name,
                }, ckpt_dir / 'hist_best.pt')
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

    model.eval()
    with torch.no_grad():
        # Generate a fresh large test set for final eval
        final_dataset = HistogramDataset(T=args.T, L=args.L, n_train=1000,
                                         n_test=5000, seed=99999, device=device)
        test_in = final_dataset.test_inputs
        test_tgt = final_dataset.test_targets

        logits = model(test_in)
        preds = logits.argmax(dim=-1)

        # Overall
        overall_acc = (preds == test_tgt).float().mean().item()

        # By count value
        results = {'overall': {'acc': overall_acc, 'n': test_in.shape[0]}}
        for c in range(args.L):
            mask = (test_tgt == c)
            if mask.any():
                acc = (preds[mask] == test_tgt[mask]).float().mean().item()
                results[f'count_{c+1}'] = {'acc': acc, 'n': mask.sum().item()}

        # Train-like vs held-out subsets
        train_mask, held_mask = filter_dataset(
            test_in, test_tgt,
            exclude_high_count=args.exclude_high_count,
            exclude_tokens=args.exclude_tokens,
        )
        if train_mask.any():
            acc = (preds[train_mask] == test_tgt[train_mask]).float().mean().item()
            results['train_like'] = {'acc': acc, 'n': train_mask.sum().item()}
        if held_mask.any():
            acc = (preds[held_mask] == test_tgt[held_mask]).float().mean().item()
            results['held_out_like'] = {'acc': acc, 'n': held_mask.sum().item()}

    for name, data in results.items():
        print(f"  {name:<20} acc={data['acc']:.1%}  (n={data['n']})")

    results_path = ckpt_dir / 'generalization_results.json'
    json_results = {
        'exp_name': exp_name,
        'variant': args.ffn_type,
        'seed': args.seed,
        'T': args.T,
        'L': args.L,
        'results': results,
    }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
