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

# Train with register tokens (inspired by "Vision Transformers Need Registers")
uv run python train_with_registers.py --num_registers 1 --num_digits 2

# Train with multiple register tokens
uv run python train_with_registers.py --num_registers 4 --num_digits 2
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_digits` | 2 | Number of input digits |
| `--num_registers` | 1 | Number of register tokens (train_with_registers.py only) |
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

## Analysis Notebooks

The project includes [Marimo](https://marimo.io/) notebooks for interpretability analysis:

```bash
# FFN activation analysis for digit overflow patterns
uv run marimo edit ffn_activation_analysis.py

# TransformerLens-based activation analysis
uv run marimo edit tl_activation_analysis.py

# Captum interpretability demo
uv run marimo edit captum_demo.py
```

## Project Structure

```
├── model/
│   ├── components.py          # MHA, FFN, GLU implementations
│   └── model.py               # OneLayerTransformer
├── train.py                   # Training script
├── train_with_registers.py    # Training with register tokens
├── ffn_activation_analysis.py # FFN activation patterns (Marimo)
├── tl_activation_analysis.py  # TransformerLens analysis (Marimo)
├── captum_demo.py             # Captum interpretability demo (Marimo)
└── checkpoints/               # Saved models
```

## Status

Work in progress. MoE implementation coming soon.
