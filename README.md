# How Mixture-of-Experts Changes Computation in Small Transformers

Investigating how replacing dense FFN with Mixture of Experts changes the computation structure, learning dynamics, and expert specialization in 1-layer transformers on algorithmic tasks.

## Guiding Research Question

**How does the choice of FFN architecture affect what the attention mechanism learns?**

We compare four architectures (matched for parameter count): standard FFN, GLU, MoE (FFN experts), and MoE-GLU. Inspired by Behrens et al. (2025) which studies the interplay between attention and FFN on counting tasks.

### Sub-questions

1. Where does computation happen (attention vs FFN), and does MoE change this balance?
2. Does MoE's routing bottleneck affect learning dynamics (grokking)?
3. Do MoE experts specialize by functional role?
4. How does GLU interact with activation-based interpretability?

## Tasks

| Task | Description | What it tests |
|------|-------------|---------------|
| **Modular Addition** | (a+b) mod 113, Nanda et al. (2023) | Grokking dynamics, Fourier analysis |
| **Add-7** | Digit-by-digit x+7 with carry propagation | Computation structure, expert specialization |
| **Histogram** | Count token frequencies, Behrens et al. (2025) | Validation of redistribution finding |

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
figures/                   # Generated figures (39 total)
checkpoints/               # Trained model checkpoints
archive/                   # Old/superseded files

train.py                   # Add-7 training script
train_modular_addition.py  # Modular addition training script
train_histogram.py         # Histogram counting training script

results_all_experiments.md # Complete experiment results and findings
paper_outline.txt          # Paper structure and figure list
experiments.md             # Original experiment plan
```

## Key Findings

See `results_all_experiments.md` for full details.

1. **MoE redistributes computation from FFN to attention** — Validated across 3 tasks. On add-7, zeroing FFN drops dense models to 9.5% but MoE retains 55%. Effect scales with task simplicity (+46pp on add-7, +15pp on modular addition, +0.7pp on histogram). Redistribution is adaptive: easy operations go to attention, hard carry propagation stays in FFN. Explained by Nanda et al.'s Fourier circuit framework — redistribution only occurs when the computation is within attention's expressive capacity.

2. **MoE's routing bottleneck forces attention to specialize** — MoE develops cleaner attention head specialization (digit-copying, carry-sensitive context) than dense models. Head-specific ablation confirms functional roles. DLA shows MoE attributes 64% of correct logits to attention (vs 40% for FFN). Activation patching provides causal confirmation.

3. **MoE accelerates grokking via regularization-routing interaction** — 2-3x faster on modular addition (5/5 seeds vs 3/5). Requires both multiple experts (E=1 fails) and hard routing (top-2 kills grokking). Effect scales with model width (1.1x at d=64, 2.7x at d=256). Dropout achieves similar speedups, but E=1 control confirms routing itself contributes.

4. **GLU hides structure, not information** — Fourier concentration drops from 0.44 to 0.07, but linear probes achieve 100% accuracy from gate, up, and product individually. The multiplicative interaction destroys per-neuron organization while preserving information in the full vector. Practical warning for neuron-level interpretability methods.

## Analysis Methods

| Method | Purpose | Script |
|--------|---------|--------|
| Component ablation | Where computation happens (zero attn/FFN) | `analysis/visualize_results.py` |
| Per-position ablation | Position-dependent computation | `analysis/visualize_results.py` |
| Carry-length stratification | Difficulty-dependent redistribution | `analysis/analyze_by_carry_length.py` |
| Direct Logit Attribution | Quantify component contributions | `analysis/dla_add7.py` |
| Attention patterns | What algorithm attention learns | `analysis/visualize_results.py` |
| Head-specific ablation | Individual head functions | `analysis/head_ablation_sorted.py` |
| Activation patching | Causal role of each component | `analysis/activation_patching.py` |
| Linear probes | Operation type decodability | `analysis/analyze_add7.py` |
| Fourier analysis | Internal representation structure | `analysis/analyze_all_variants.py` |
| GLU gate probes | How GLU hides structure | `analysis/glu_gate_probes.py` |
| Expert routing/ablation | Expert specialization | `analysis/analyze_add7.py` |

## References

- Nanda et al. (2023) "Progress measures for grokking via mechanistic interpretability"
- Quirke & Barez (2024) "Understanding Addition in Transformers"
- Behrens et al. (2025) "Counting in Small Transformers: The Delicate Interplay between Attention and Feed-Forward Layers"
- Yuan et al. (2025) "Is Random Attention Sufficient for Sequence Modeling?"
