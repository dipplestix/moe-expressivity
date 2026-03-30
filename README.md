# How Mixture-of-Experts Changes Computation in Small Transformers

Investigating how replacing dense FFN with Mixture of Experts changes the computation structure, learning dynamics, and expert specialization in 1-layer transformers on algorithmic tasks.

## Research Questions

1. How does MoE change where computation happens (attention vs FFN)?
2. Does MoE's routing bottleneck affect learning dynamics (grokking)?
3. Do experts specialize by functional role?
4. How does GLU interact with MoE and interpretability?

## Tasks

| Task | Description | What it tests |
|------|-------------|---------------|
| **Modular Addition** | (a+b) mod 113, Nanda et al. (2023) | Grokking dynamics, Fourier analysis |
| **Add-7** | Digit-by-digit x+7 with carry propagation | Computation structure, expert specialization |
| **Histogram** | Count token frequencies in a sequence, Glorot et al. (2025) | Validation of computation redistribution |

## Architecture Variants

Four FFN variants compared in a 1-layer transformer (embedding + attention + FFN + output):

| Variant | Description |
|---------|-------------|
| FFN | Standard dense feed-forward (GELU activation) |
| GLU | Gated Linear Unit (SiLU gate x linear projection) |
| MoE | 4 FFN experts with top-1 routing + load-balancing loss |
| MoE-GLU | 4 GLU experts with top-1 routing + load-balancing loss |

## Installation

```bash
uv sync
```

## Quick Start

```bash
# Train one model
.venv/bin/python train.py --ffn_type moe --num_digits 3 --no_wandb

# Run all experiments for a task (see scripts/README.md for full list)
bash scripts/run_multiseed.sh

# Generate figures (see analysis/README.md)
.venv/bin/python analysis/visualize_results.py
```

## Project Structure

```
model/                     # Architecture code (see model/README.md)
formerlens/                # TransformerLens-compatible hooked models (see formerlens/README.md)
data/                      # Dataset classes (see data/README.md)
scripts/                   # Experiment runner scripts (see scripts/README.md)
analysis/                  # Analysis and visualization (see analysis/README.md)
figures/                   # Generated figures
checkpoints/               # Trained model checkpoints

train.py                   # Add-7 training script
train_modular_addition.py  # Modular addition training script
train_histogram.py         # Histogram counting training script

results_all_experiments.md # Complete experiment results and findings
experiments.md             # Original experiment plan
```

## Key Findings

See `results_all_experiments.md` for full details.

1. **MoE redistributes computation from FFN to attention** — On add-7, zeroing FFN drops dense models to 9.5% but MoE retains 55%. Per-position analysis shows MoE pushes easy operations into attention.
2. **MoE accelerates grokking** — 2-3x faster on modular addition, requiring both multiple experts (E=1 fails) and hard routing (top-2 fails). Effect scales with model width.
3. **GLU hides internal structure** — Fourier concentration drops from 0.44 to 0.07, despite same accuracy. Interpretability warning for activation-based methods.
4. **MoE-GLU experts partially specialize by operation type** — Routing aligns with +7/+1/+0 operations on add-7, confirmed by expert ablation.

## References

- Nanda et al. (2023) "Progress measures for grokking via mechanistic interpretability"
- Quirke & Barez (2024) "Understanding Addition in Transformers"
- Glorot et al. (2025) "Counting in Small Transformers: The Delicate Interplay between Attention and Feed-Forward Layers"
