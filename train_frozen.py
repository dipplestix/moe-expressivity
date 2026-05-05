"""
Frozen component training on modular addition.

Train with one component frozen at random initialization:
  - Frozen attention: attention weights locked, only FFN learns
  - Frozen FFN: FFN weights locked, only attention learns

Tests whether one component can compensate for the other.
If frozen-attention still works, FFN can solve it alone.
If frozen-FFN fails, FFN is essential.
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

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = ModularAdditionDataset(p=args.p, train_frac=args.train_frac, seed=args.seed, device=device)
    print(f"p={args.p}, vocab={dataset.vocab_size}, train={len(dataset.train_inputs)}, test={len(dataset.test_inputs)}")

    model = OneLayerTransformer(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ffn_type=args.ffn_type,
        vocab_size=dataset.vocab_size,
        max_seq_len=3,
        use_norm=False,
        is_causal=False,
        tie_embeddings=False,
        activation=args.activation,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
    ).to(device)

    # Freeze the specified component
    if args.freeze == "attention":
        for name, param in model.named_parameters():
            if 'atn' in name:
                param.requires_grad = False
        print("Frozen: attention weights")
    elif args.freeze == "ffn":
        for name, param in model.named_parameters():
            if 'ffn' in name:
                param.requires_grad = False
        print("Frozen: FFN weights")
    else:
        print("No component frozen (full training)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(dataset.train_inputs)
        loss = F.cross_entropy(logits[:, 2, :], dataset.train_targets)

        if args.ffn_type in ("moe", "moe_glu") and args.freeze != "ffn":
            loss = loss + 0.01 * model._aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_preds = model(dataset.train_inputs)[:, 2, :].argmax(dim=-1)
                train_acc = (train_preds == dataset.train_targets).float().mean().item()

                test_preds = model(dataset.test_inputs)[:, 2, :].argmax(dim=-1)
                test_acc = (test_preds == dataset.test_targets).float().mean().item()

            print(f"Epoch {epoch}: loss={loss.item():.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_acc": test_acc,
                    "freeze": args.freeze,
                    "config": {
                        "model_dim": args.model_dim,
                        "num_heads": args.num_heads,
                        "ffn_type": args.ffn_type,
                        "vocab_size": args.p + 1,
                        "max_seq_len": 3,
                        "use_norm": False,
                        "is_causal": False,
                        "tie_embeddings": False,
                        "activation": args.activation,
                        "intermediate_dim": args.intermediate_dim,
                        "num_experts": args.num_experts,
                        "top_k": args.top_k,
                    },
                }, ckpt_dir / "modadd_best.pt")

    print(f"\nDone. Best test acc: {best_test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with frozen component on modular addition")

    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", type=str, default="none", choices=["none", "attention", "ffn"])

    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_type", type=str, default="ffn", choices=["ffn", "glu", "moe", "moe_glu"])
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--intermediate_dim", type=int, default=512)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40000)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    train(parser.parse_args())
