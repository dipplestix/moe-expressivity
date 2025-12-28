import torch
import torch.nn.functional as F
import wandb
import argparse
import sys
from pathlib import Path
sys.path.insert(0, 'model')
from model import OneLayerTransformer


# Token definitions
PAD_TOKEN = 10
EOS_TOKEN = 11
VOCAB_SIZE = 12  # 0-9 + PAD + EOS


def num_to_reversed_digits(n, num_digits):
    """Convert number to reversed digit list, padded to num_digits."""
    digits = []
    for _ in range(num_digits):
        digits.append(n % 10)
        n //= 10
    return digits


def generate_batch(batch_size, num_digits, device):
    """Generate a batch of add-7 examples.

    Format: <reversed_input><EOS><reversed_output><EOS>
    """
    max_val = 10 ** num_digits - 1

    # Random input numbers
    inputs = torch.randint(0, max_val + 1, (batch_size,))
    outputs = inputs + 7

    # Output may have one more digit due to carry
    out_num_digits = num_digits + 1

    sequences = []
    for i in range(batch_size):
        in_digits = num_to_reversed_digits(inputs[i].item(), num_digits)
        out_digits = num_to_reversed_digits(outputs[i].item(), out_num_digits)

        # Sequence: input_digits + EOS + output_digits + EOS
        seq = in_digits + [EOS_TOKEN] + out_digits + [EOS_TOKEN]
        sequences.append(seq)

    # Stack into tensor
    sequences = torch.tensor(sequences, dtype=torch.long, device=device)
    return sequences, num_digits  # return input length for masking


def compute_loss(model, sequences, input_len):
    """Compute cross-entropy loss on output portion only."""
    # Input: all tokens except last
    # Target: all tokens except first
    x = sequences[:, :-1]
    targets = sequences[:, 1:]

    logits = model(x)

    # Create mask: only compute loss on output portion (after first EOS)
    # The first EOS is at position input_len, so we predict from input_len onwards
    mask = torch.zeros_like(targets, dtype=torch.float)
    mask[:, input_len:] = 1.0

    # Flatten for cross-entropy
    B, T = targets.shape
    logits_flat = logits.reshape(B * T, -1)
    targets_flat = targets.reshape(B * T)
    mask_flat = mask.reshape(B * T)

    # Compute per-token loss
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')

    # Apply mask and average
    loss = (loss_per_token * mask_flat).sum() / mask_flat.sum()

    return loss


def evaluate(model, num_digits, device, num_samples=256):
    """Evaluate exact-match accuracy on random samples."""
    model.eval()

    max_val = 10 ** num_digits - 1
    out_num_digits = num_digits + 1

    correct = 0
    total = num_samples

    with torch.no_grad():
        for _ in range(num_samples):
            # Generate single example
            n = torch.randint(0, max_val + 1, (1,)).item()
            expected = n + 7

            in_digits = num_to_reversed_digits(n, num_digits)
            expected_out = num_to_reversed_digits(expected, out_num_digits)

            # Build input sequence (just the input + EOS)
            seq = torch.tensor([in_digits + [EOS_TOKEN]], dtype=torch.long, device=device)

            # Autoregressive generation
            generated = []
            for _ in range(out_num_digits + 1):  # +1 for final EOS
                logits = model(seq)
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)

                if next_token == EOS_TOKEN:
                    break

                seq = torch.cat([seq, torch.tensor([[next_token]], device=device)], dim=1)

            # Check if output matches (excluding final EOS)
            pred_digits = [t for t in generated if t != EOS_TOKEN]
            if pred_digits == expected_out:
                correct += 1

    model.train()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_digits', type=int, default=2)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ffn_type', type=str, default='ffn', choices=['ffn', 'glu'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5, help='Stop if no improvement for N evals')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project', type=str, default='add-7-transformer')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--use_norm', dest='use_norm', action='store_true', default=True, help='Enable RMSNorm layers')
    parser.add_argument('--no_use_norm', dest='use_norm', action='store_false', help='Disable RMSNorm layers')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Create model
    model = OneLayerTransformer(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ffn_type=args.ffn_type,
        vocab_size=VOCAB_SIZE,
        use_norm=args.use_norm,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # Early stopping tracking
    best_acc = 0.0
    patience_counter = 0

    # Training loop
    model.train()
    for step in range(1, args.steps + 1):
        sequences, input_len = generate_batch(args.batch_size, args.num_digits, device)

        optimizer.zero_grad()
        loss = compute_loss(model, sequences, input_len)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % 10 == 0:
            log_dict = {'loss': loss.item(), 'step': step}

            if step % args.eval_interval == 0:
                acc = evaluate(model, args.num_digits, device)
                log_dict['accuracy'] = acc
                print(f"Step {step}: loss={loss.item():.4f}, accuracy={acc:.2%}")

                # Early stopping check
                if acc > best_acc:
                    best_acc = acc
                    patience_counter = 0
                    # Save best model
                    ckpt_path = ckpt_dir / 'best_model.pt'
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': acc,
                        'config': vars(args),
                    }, ckpt_path)
                    print(f"Saved best model to {ckpt_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"Early stopping: no improvement for {args.patience} evals")
                        break
            else:
                print(f"Step {step}: loss={loss.item():.4f}")

            if not args.no_wandb:
                wandb.log(log_dict)

    # Final evaluation
    final_acc = evaluate(model, args.num_digits, device, num_samples=1000)
    print(f"\nFinal accuracy: {final_acc:.2%}")

    if not args.no_wandb:
        wandb.log({'final_accuracy': final_acc})
        wandb.finish()


if __name__ == '__main__':
    main()
