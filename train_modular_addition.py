"""
Training script for (a+b) mod p with a 1-layer transformer.

Replicates Nanda et al. (2023) grokking setup:
  - Full-batch training on 30% of p^2 pairs, eval on held-out 70%
  - d_model=128, num_heads=4, d_mlp=512, p=113
  - AdamW lr=1e-3, weight_decay=1.0, 40k epochs
  - Loss: cross-entropy at position 2 (the '=' token)
  - Expected: train memorizes early (~500 epochs), test groks late (~10k-30k)

MoE variant: --ffn_type moe --num_experts 4 --top_k 1
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "model")
from model import OneLayerTransformer
from data.modular_addition import ModularAdditionDataset

try:
    import wandb
except ImportError:
    wandb = None


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set all random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Dataset
    dataset = ModularAdditionDataset(p=args.p, train_frac=args.train_frac, seed=args.seed, device=device)
    print(f"p={args.p}, vocab={dataset.vocab_size}, train={len(dataset.train_inputs)}, test={len(dataset.test_inputs)}")

    # Model
    model = OneLayerTransformer(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ffn_type=args.ffn_type,
        dropout=args.dropout,
        vocab_size=dataset.vocab_size,
        max_seq_len=3,
        use_norm=args.use_norm,
        is_causal=False,
        tie_embeddings=False,
        activation=args.activation,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Wandb
    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    best_test_acc = 0.0
    milestones = {50: False, 90: False, 99: False}

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(dataset.train_inputs)  # (N, 3, vocab)
        loss = F.cross_entropy(logits[:, 2, :], dataset.train_targets)

        if args.ffn_type in ("moe", "moe_glu"):
            aux_loss = model._aux_loss
            loss = loss + args.moe_balance_coeff * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Train accuracy
                train_logits = model(dataset.train_inputs)
                train_preds = train_logits[:, 2, :].argmax(dim=-1)
                train_acc = (train_preds == dataset.train_targets).float().mean().item()

                # Test accuracy
                test_logits = model(dataset.test_inputs)
                test_preds = test_logits[:, 2, :].argmax(dim=-1)
                test_acc = (test_preds == dataset.test_targets).float().mean().item()

            log_dict = {
                "epoch": epoch,
                "loss": loss.item(),
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
            if args.ffn_type in ("moe", "moe_glu"):
                log_dict["aux_loss"] = aux_loss.item()

            print(f"Epoch {epoch}: loss={loss.item():.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

            if use_wandb:
                wandb.log(log_dict)

            #Milestone checkpoints
            for pct, saved in milestones.items():
                if not saved and test_acc > pct / 100.0:
                    path = ckpt_dir / f"modadd_test{pct}.pt"
                    _save_checkpoint(model, optimizer, epoch, test_acc, args, path)
                    print(f"  Milestone: test_acc > {pct}%, saved {path}")
                    milestones[pct] = True

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                _save_checkpoint(model, optimizer, epoch, test_acc, args, ckpt_dir / "modadd_best.pt")

        if epoch % args.save_interval == 0:
            _save_checkpoint(model, optimizer, epoch, best_test_acc, args, ckpt_dir / f"modadd_epoch{epoch}.pt")

    print(f"\nDone. Best test acc: {best_test_acc:.4f}")
    if use_wandb:
        wandb.finish()


def _save_checkpoint(model, optimizer, epoch, test_acc, args, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_acc": test_acc,
        "config": {
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "ffn_type": args.ffn_type,
            "vocab_size": args.p + 1,
            "max_seq_len": 3,
            "use_norm": args.use_norm,
            "is_causal": False,
            "tie_embeddings": False,
            "activation": args.activation,
            "dropout": args.dropout,
            "intermediate_dim": args.intermediate_dim,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
        },
    }, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1-layer transformer on modular addition")

    # Task
    parser.add_argument("--p", type=int, default=113, help="Prime modulus")
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_type", type=str, default="ffn", choices=["ffn", "glu", "moe", "moe_glu"])
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu", "relu"])
    parser.add_argument("--intermediate_dim", type=int, default=512)
    parser.add_argument("--use_norm", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40000)
    parser.add_argument("--moe_balance_coeff", type=float, default=0.01)

    # Logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="modular-addition")
    parser.add_argument("--no_wandb", action="store_true")

    train(parser.parse_args())
