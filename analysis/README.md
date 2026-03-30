# analysis/

Scripts for analyzing trained models and generating publication figures. All scripts auto-detect the project root so they can be run from any directory.

## How to Run

```bash
.venv/bin/python analysis/<script_name>.py
```

## Main Scripts

### `visualize_results.py`

Generates all publication-quality figures. Reads directly from checkpoints — no hardcoded values.

**Color scheme**: FFN=blue, GLU=orange, MoE=green, MoE-GLU=red (consistent across all figures).

Main figures:
| Figure | Description | Data source |
|--------|-------------|-------------|
| Fig 1 | Grokking timeline (modular addition, 5 seeds, shaded std) | `modadd_*_s*/modadd_epoch*.pt` |
| Fig 2 | Regularization baselines (grokking speed + reliability) | `modadd_ffn_drop*/modadd_ffn_wd*` |
| Fig 3 | Number of experts scaling (E=1..16) | `modadd_moe_e*_s*` |
| Fig 4 | H1 component ablation, add-7 (no norm) | `add7_*_nonorm_s*` |
| Fig 5 | H3 expert-operation routing heatmap (MoE-GLU, seed 42) | `add7_moe_glu_nonorm_s42` |
| Fig 6 | Neuron Fourier concentration histograms | `modadd_*_s42` |
| Fig 7 | Model width scaling (d=64,128,256) | `modadd_*_d*_s*` |
| Fig 8 | H1 component ablation, modular addition (no norm) | `modadd_*_s*` |
| Fig 9 | Fourier structure over training (seed 42) | `modadd_*_s42/modadd_epoch*.pt` |
| Fig 10 | Per-position accuracy under ablation (add-7, no norm) | `add7_*_nonorm_s*` |

Appendix figures:
| Figure | Description |
|--------|-------------|
| Fig A1 | Norm vs no-norm grokking comparison |
| Fig A2 | Top-k routing (top-1 vs top-2 accuracy) |
| Fig A3 | Norm effect on component ablation (add-7) |
| Fig A4 | Per-seed expert routing variability (5 seeds x 2 MoE variants) |

Output: `figures/fig*.png`, `figures/figa*.png`

### `analyze_add7.py`

Full H1-H3 analysis on add-7 across all 4 variants x 5 seeds:
- H1: Component ablation (zero attention / zero FFN)
- H1: Linear probes predicting operation type (+7/+1/+0) from attention/FFN outputs
- H3: Routing mutual information between expert selection and operation type (MoE only)
- H3: Expert ablation — per-operation accuracy drops when each expert is removed

### `analyze_all_variants.py`

Fourier analysis on modular addition:
- Per-neuron Fourier concentration (how much each neuron responds to a single frequency)
- Router Fourier concentration (how clean the routing signal is)
- Expert frequency specialization (do different experts prefer different frequencies?)
- Grokking timeline from milestone checkpoints

### `analyze_modadd_ablation.py`

Component ablation (zero attention / zero FFN) on modular addition across all 4 variants x 5 seeds. Compares norm vs no-norm settings.

## Legacy Scripts

These are from earlier exploration and not used in the paper:
- `ffn_activation_analysis.py` — Marimo notebook for FFN activation patterns
- `tl_activation_analysis.py` — Marimo notebook for TransformerLens analysis
- `captum_demo.py` — Marimo notebook for Captum interpretability demo
- `attention_analysis.ipynb` — Jupyter notebook for attention patterns
- `ffn_analysis.ipynb` — Jupyter notebook for FFN analysis
