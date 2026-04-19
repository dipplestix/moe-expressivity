"""
Frozen component training on add-7.

Mirrors train_frozen.py (modular addition) but uses the add-7 data/loss from
train.py. Trains with one component frozen at random initialization:
  - Frozen attention: attention weights locked, only FFN learns
  - Frozen FFN: FFN weights locked, only attention learns
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, "model")
from model import OneLayerTransformer
from train import VOCAB_SIZE, generate_batch, compute_loss, evaluate


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    best_acc = 0.0
    patience_counter = 0
    model.train()

    for step in range(1, args.steps + 1):
        sequences, input_len = generate_batch(args.batch_size, args.num_digits, device)

        optimizer.zero_grad()
        loss = compute_loss(model, sequences, input_len)
        if args.ffn_type in ("moe", "moe_glu") and args.freeze != "ffn":
            loss = loss + args.moe_balance_coeff * model._aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
        )
        optimizer.step()

        if step % args.eval_interval == 0:
            acc = evaluate(model, args.num_digits, device)
            print(f"Step {step}: loss={loss.item():.4f} acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "accuracy": acc,
                        "freeze": args.freeze,
                        "config": {
                            "model_dim": args.model_dim,
                            "num_heads": args.num_heads,
                            "ffn_type": args.ffn_type,
                            "vocab_size": VOCAB_SIZE,
                            "use_norm": args.use_norm,
                            "num_experts": args.num_experts,
                            "top_k": args.top_k,
                        },
                        "args": vars(args),
                    },
                    ckpt_dir / "best_model.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping after {args.patience} evals without improvement")
                    break

    final_acc = evaluate(model, args.num_digits, device, num_samples=1000)
    print(f"\nDone. Best acc: {best_acc:.4f}, Final acc: {final_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frozen component training on add-7")
    parser.add_argument("--num_digits", type=int, default=3)
    parser.add_argument("--freeze", type=str, default="none", choices=["none", "attention", "ffn"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_type", type=str, default="ffn", choices=["ffn", "glu", "moe", "moe_glu"])
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--moe_balance_coeff", type=float, default=0.01)
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu", "relu"])

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    parser.add_argument("--use_norm", dest="use_norm", action="store_true", default=True)
    parser.add_argument("--no_use_norm", dest="use_norm", action="store_false")

    train(parser.parse_args())
