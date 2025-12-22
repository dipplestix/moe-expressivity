# MoE Expressivity

Investigating how different FFN architectures affect what attention learns in a minimal transformer setting.

## Motivation

We train a one-layer transformer to add 7 to a number (represented digit-by-digit in reversed order). This task has a simple, well-defined structure:

1. **First digit:** Add 7 (may overflow)
2. **Second digit:** Add 1 if first digit overflows (>=10)
3. **All other digits:** Add 1 if there's a carry chain (all preceding digits were 9 and overflowed)

Example: `593 + 7 = 600`
- Input (reversed): `3 9 5`
- Output (reversed): `0 0 6`
- Pattern: 3+7=10 (overflow) -> 9+1=10 (overflow) -> 5+1=6

## Research Question

How does the choice of FFN architecture affect what the attention mechanism learns?

We compare three architectures (matched for parameter count):
1. **Standard FFN** - Dense feedforward with 4x hidden dimension
2. **GLU** - Gated Linear Unit
3. **MoE** - Mixture of Experts (with both FFN and GLU variants)

Inspired by [Yuan et al. (2025)](https://arxiv.org/abs/2506.01115) which analyzes attention patterns in arithmetic tasks.

## Installation

```bash
uv sync
```

## Usage

```bash
# Train with standard FFN
uv run python train.py --ffn_type ffn --num_digits 2

# Train with GLU
uv run python train.py --ffn_type glu --num_digits 2

# Disable wandb for local testing
uv run python train.py --no_wandb --patience 3
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_digits` | 2 | Number of input digits |
| `--model_dim` | 64 | Model dimension |
| `--num_heads` | 4 | Number of attention heads |
| `--ffn_type` | ffn | FFN type: `ffn` or `glu` |
| `--batch_size` | 128 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--max_grad_norm` | 1.0 | Gradient clipping |
| `--steps` | 5000 | Max training steps |
| `--eval_interval` | 100 | Evaluation frequency |
| `--patience` | 5 | Early stopping patience |
| `--checkpoint_dir` | checkpoints | Checkpoint directory |
| `--no_wandb` | False | Disable wandb logging |

## Project Structure

```
├── model/
│   ├── components.py   # MHA, FFN, GLU implementations
│   └── model.py        # OneLayerTransformer
├── train.py            # Training script
└── checkpoints/        # Saved models
```

## Status

Work in progress. MoE implementation coming soon.
