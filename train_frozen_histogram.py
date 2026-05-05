"""
Frozen component training on histogram counting.

Mirrors train_frozen_add7.py but uses histogram data/loss from train_histogram.py.
Trains with one component frozen at random initialization:
  - Frozen attention: attention weights locked, only FFN learns
  - Frozen FFN: FFN weights locked, only attention learns
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "model")
from model import OneLayerTransformer
from data.histogram import HistogramDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = HistogramDataset(
        T=args.T, L=args.L, n_train=args.n_train, n_test=args.n_test,
        seed=args.seed, device=device,
    )
    print(f"T={args.T}, L={args.L}, vocab={dataset.vocab_size}, "
          f"classes={dataset.num_classes}, train={len(dataset.train_inputs)}, "
          f"test={len(dataset.test_inputs)}")

    model = OneLayerTransformer(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ffn_type=args.ffn_type,
        dropout=args.dropout,
        vocab_size=dataset.vocab_size,
        max_seq_len=args.L,
        use_norm=args.use_norm,
        is_causal=False,
        tie_embeddings=False,
        activation=args.activation,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
    ).to(device)

    # Replace output head: model outputs vocab logits but we need count classes
    model.unembed = nn.Linear(args.model_dim, dataset.num_classes, bias=False).to(device)

    # Freeze the specified component
    if args.freeze == "attention":
        for name, param in model.named_parameters():
            if "atn" in name:
                param.requires_grad = False
        print("Frozen: attention weights")
    elif args.freeze == "ffn":
        for name, param in model.named_parameters():
            if "ffn" in name:
                param.requires_grad = False
        print("Frozen: FFN weights")
    else:
        print("No component frozen (full training)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    best_test_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(dataset.train_inputs)
        loss = F.cross_entropy(
            logits.view(-1, dataset.num_classes),
            dataset.train_targets.view(-1),
        )

        if args.ffn_type in ("moe", "moe_glu") and args.freeze != "ffn":
            loss = loss + args.moe_balance_coeff * model._aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        if epoch % args.log_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_logits = model(dataset.train_inputs)
                train_preds = train_logits.argmax(dim=-1)
                train_acc = (train_preds == dataset.train_targets).float().mean().item()

                test_logits = model(dataset.test_inputs)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == dataset.test_targets).float().mean().item()

            print(f"Epoch {epoch}: loss={loss.item():.4f} "
                  f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "accuracy": test_acc,
                    "freeze": args.freeze,
                    "config": {
                        "model_dim": args.model_dim,
                        "num_heads": args.num_heads,
                        "ffn_type": args.ffn_type,
                        "vocab_size": dataset.vocab_size,
                        "max_seq_len": args.L,
                        "use_norm": args.use_norm,
                        "is_causal": False,
                        "tie_embeddings": False,
                        "activation": args.activation,
                        "intermediate_dim": args.intermediate_dim,
                        "num_experts": args.num_experts,
                        "top_k": args.top_k,
                        "T": args.T,
                        "L": args.L,
                        "num_classes": dataset.num_classes,
                    },
                    "args": vars(args),
                }, ckpt_dir / "hist_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping after {args.patience} evals without improvement")
                    break

        if epoch % args.refresh_interval == 0:
            dataset = HistogramDataset(
                T=args.T, L=args.L, n_train=args.n_train,
                n_test=args.n_test, seed=args.seed + epoch, device=device,
            )

    print(f"\nDone. Best test acc: {best_test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frozen component training on histogram")

    # Task
    parser.add_argument("--T", type=int, default=32, help="Alphabet size")
    parser.add_argument("--L", type=int, default=10, help="Sequence length")
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh_interval", type=int, default=50)

    # Freeze
    parser.add_argument("--freeze", type=str, default="none",
                        choices=["none", "attention", "ffn"])

    # Model
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_type", type=str, default="ffn",
                        choices=["ffn", "glu", "moe", "moe_glu"])
    parser.add_argument("--activation", type=str, default="gelu",
                        choices=["gelu", "silu", "relu"])
    parser.add_argument("--intermediate_dim", type=int, default=512)
    parser.add_argument("--use_norm", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--moe_balance_coeff", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=50)

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--no_wandb", action="store_true")

    train(parser.parse_args())
