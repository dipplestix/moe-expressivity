# MoE Expressivity

Investigating how different FFN architectures affect what attention learns in a minimal transformer setting.

## The Add-7 Task

We train a one-layer transformer to add 7 to a number (represented digit-by-digit in reversed order). This task has a simple, well-defined structure:

1. **First digit:** Add 7 (may overflow)
2. **Second digit:** Add 1 if first digit overflows (>=10)
3. **All other digits:** Add 1 if there's a carry chain (all preceding digits were 9 and overflowed)

**Example:** `593 + 7 = 600`
```
Input (reversed):  3 9 5
Output (reversed): 0 0 6
Pattern: 3+7=10 (overflow) -> 9+1=10 (overflow) -> 5+1=6
```

## Research Question

How does the choice of FFN architecture affect what the attention mechanism learns?

We compare three architectures:
- **Standard FFN** - Dense feedforward with 4x hidden dimension
- **GLU** - Gated Linear Unit (SwiGLU-style, no bias)
- **MoE** - Mixture of Experts *(coming soon)*

Inspired by [Yuan et al. (2025)](https://arxiv.org/abs/2506.01115) which analyzes attention patterns in arithmetic tasks.

## Architecture

The `OneLayerTransformer` consists of:
- Token + positional embeddings
- Pre-norm multi-head attention (RMSNorm)
- Pre-norm FFN/GLU
- Tied output embeddings (weight sharing with input)

Key features:
- **`use_norm`** flag to enable/disable RMSNorm layers
- **Causal attention** by default
- **SiLU activation** in FFN/GLU

Run `uv run python count_params.py` to see parameter breakdown by component.

## Installation

```bash
uv sync
```

**Requirements:** Python 3.13+, PyTorch 2.9+

## Training

### Basic Training

```bash
# Train with standard FFN
uv run python train.py --ffn_type ffn --num_digits 2

# Train with GLU
uv run python train.py --ffn_type glu --num_digits 2

# Local testing (no wandb)
uv run python train.py --no_wandb --patience 3
```

### Register Tokens

Based on [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) (Darcet et al., 2024), register tokens provide a "scratch space" for attention.

```bash
# Train with 1 register token
uv run python train_with_registers.py --num_registers 1 --num_digits 2

# Train with multiple registers
uv run python train_with_registers.py --num_registers 4 --num_digits 2
```

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_digits` | 2 | Number of input digits |
| `--num_registers` | 1 | Register tokens (train_with_registers.py) |
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

## Interpretability

### FormerLens

A TransformerLens-compatible library for capturing intermediate activations:

```python
from formerlens import HookedOneLayerTransformer

# Load from checkpoint
model = HookedOneLayerTransformer.from_checkpoint("checkpoints/best_model.pt")

# Run with activation cache
logits, cache = model.run_with_cache(tokens)

# Access activations
attn_pattern = cache["atn.hook_pattern"]
ffn_activations = cache["ffn.hook_post_act"]
```

### Analysis Notebooks

[Marimo](https://marimo.io/) notebooks for interactive analysis:

```bash
# FFN activation analysis - digit overflow patterns
uv run marimo edit ffn_activation_analysis.py

# TransformerLens-based activation analysis
uv run marimo edit tl_activation_analysis.py

# Captum interpretability demo
uv run marimo edit captum_demo.py
```

Jupyter notebook: `attention_analysis.ipynb`

## Project Structure

```
moe-expressivity/
├── model/
│   ├── components.py           # MHA, FFN, GLU implementations
│   └── model.py                # OneLayerTransformer
├── formerlens/
│   ├── __init__.py             # Public API
│   ├── hooked_components.py    # Hooked MHA, FFN, GLU
│   └── hooked_former.py        # HookedOneLayerTransformer
├── train.py                    # Basic training script
├── train_with_registers.py     # Training with register tokens
├── count_params.py             # Parameter counting utility
├── ffn_activation_analysis.py  # Marimo: FFN analysis
├── tl_activation_analysis.py   # Marimo: TransformerLens analysis
├── captum_demo.py              # Marimo: Captum demo
├── attention_analysis.ipynb    # Jupyter: attention patterns
└── checkpoints/                # Saved models
```

## Dependencies

- **torch** - Core deep learning
- **wandb** - Experiment tracking
- **transformer-lens** - Mechanistic interpretability
- **captum** - PyTorch interpretability
- **marimo** - Reactive notebooks
- **matplotlib** - Visualization
- **circuitsvis** - Attention visualization

## Status

Work in progress. MoE implementation coming soon.
