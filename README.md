# Sparsity Moves Computation: How FFN Architecture Reshapes Attention in Small Transformers

Code for studying how the choice of FFN architecture (dense FFN, GLU, MoE, MoE-GLU) shapes computation in 1-layer transformers on three algorithmic tasks: add-7, modular addition, and histogram counting.

## Installation

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

All commands below assume the project root as the working directory and use `uv run python` (or `bash` for shell scripts). Substituting `.venv/bin/python` works equivalently.

## Reproducing the paper end-to-end

The full pipeline is **(1) get checkpoints → (2) regenerate figures**. Training is optional; pre-trained checkpoints for every condition reported in the paper are hosted on the Hugging Face Hub and can be downloaded in one command.

### 1. Get checkpoints

**Option A — download pre-trained checkpoints (recommended).** All checkpoints used in the paper (5 seeds per condition, every task / architecture / routing variant) are hosted anonymously on the Hugging Face Hub:

> <https://huggingface.co/Sparsity-Moves-Computation/moe-redistribution-checkpoints>

The repository is public and can be downloaded without an account.

```bash
hf download Sparsity-Moves-Computation/moe-redistribution-checkpoints \
    --repo-type model \
    --local-dir checkpoints/
```

(Older `huggingface-cli` syntax: `huggingface-cli download Sparsity-Moves-Computation/moe-redistribution-checkpoints --local-dir checkpoints/`.)

The download is resumable; rerunning the same command skips files already present.

To grab a single checkpoint from a script:

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download(
    repo_id="Sparsity-Moves-Computation/moe-redistribution-checkpoints",
    filename="add7_ffn_nonorm_s42/best_model.pt",
)
ck = torch.load(path, weights_only=False, map_location="cpu")
print(ck["config"], ck.get("accuracy"))
```

**Option B — train from scratch.** Each shell script below sweeps the relevant variants across seeds 42, 137, 256, 512, 1024 and writes to `checkpoints/<run_name>/`. Training scripts skip runs whose best checkpoint already exists, so reruns are safe. Pass `--no_wandb` (already set in the shell scripts) to disable wandb.

The three core sweeps reproduce all main-text figures:

```bash
bash scripts/run_multiseed.sh             # modular addition, 4 variants x 5 seeds
bash scripts/run_add7_multiseed.sh        # add-7,            4 variants x 5 seeds
bash scripts/run_histogram_multiseed.sh   # histogram,        4 variants x 5 seeds
```

Appendix experiments (regularization, expert count, model width, top-k, normalization, parameter matching, random routing, generalization, frozen components):

```bash
# Modular addition controls
bash scripts/run_exp1_regularization_baselines.sh
bash scripts/run_exp2_num_experts.sh
bash scripts/run_exp3_model_width.sh
bash scripts/run_exp5_top_k.sh
bash scripts/run_modadd_norm.sh
bash scripts/run_param_matched_glu.sh
bash scripts/run_param_matched_moe_glu.sh
bash scripts/run_modadd_per_active_match_moe_glu.sh
bash scripts/run_modadd_silu_symmetry.sh
bash scripts/run_modadd_d340_silu.sh
bash scripts/run_random_routing.sh

# Add-7 controls
bash scripts/run_add7_nonorm.sh
bash scripts/run_add7_param_matched_glu.sh
bash scripts/run_add7_param_matched_moe_glu.sh
bash scripts/run_add7_per_active_match_moe_glu.sh
bash scripts/run_add7_h170_both_activations.sh
bash scripts/run_add7_randroute.sh
bash scripts/run_frozen_components.sh

# Histogram controls
bash scripts/run_hist_exp1_regularization.sh
bash scripts/run_hist_exp2_num_experts.sh
bash scripts/run_hist_exp3_width.sh
bash scripts/run_hist_exp5_topk.sh
bash scripts/run_hist_param_matched.sh
bash scripts/run_hist_per_active_match_moe_glu.sh
bash scripts/run_histogram_controls.sh
bash scripts/run_frozen_histogram.sh

# Generalization (out-of-distribution carry-length / digit-count)
bash scripts/run_generalization.sh

# Activation symmetry (SiLU vs GELU, all tasks)
bash scripts/run_activation_symmetry_overnight.sh
```

See `scripts/README.md` for per-script run counts and expected wall-clock times.

To train a single model directly:

```bash
uv run python train.py --ffn_type moe --num_digits 3 --no_wandb
uv run python train_modular_addition.py --ffn_type glu --no_wandb
uv run python train_histogram.py --ffn_type moe_glu --no_wandb
```

### 2. Regenerate figures

All analysis scripts read directly from `checkpoints/`, autodetect the project root, and write to `figures/`. Run them from the project root:

```bash
# Main text figures
uv run python analysis/visualize_results.py             # Figs 1-12 (grokking timeline, ablations, routing, etc.)
uv run python analysis/make_fig1_headline.py            # Fig 1 headline (matched-axes version)
uv run python analysis/make_matched_ablation_figures.py # Parameter-matched component ablation
uv run python analysis/make_matched_figures.py          # Matched per-position + routing
uv run python analysis/make_matched_fourier_figure.py   # Fourier concentration histograms

# Causal / mechanistic analyses
uv run python analysis/activation_patching.py           # Add-7 activation patching flip rates
uv run python analysis/dla_add7.py                      # Direct logit attribution by position / carry length
uv run python analysis/head_ablation_sorted.py          # Per-head ablation, sorted by function
uv run python analysis/analyze_by_carry_length.py       # Stratified by carry-chain length

# Architecture-specific
uv run python analysis/glu_gate_probes.py               # GLU gate / up-proj / product probes
uv run python analysis/glu_decomposition_figure.py      # GLU per-neuron Fourier destruction
uv run python analysis/glu_weight_decomposition.py      # Pearce et al. bilinear tensor decomposition
uv run python analysis/analyze_all_variants.py          # Modadd Fourier across variants
uv run python analysis/weight_norm_tracking.py          # Weight norm during grokking

# Add-7 / modadd / histogram top-level rollups
uv run python analysis/analyze_add7.py
uv run python analysis/analyze_modadd_ablation.py
uv run python analysis/analyze_histogram_ablation.py
uv run python analysis/analyze_phase123.py

# Activation symmetry, generalization, reviewer follow-ups
uv run python analysis/analyze_activation_symmetry.py
uv run python analysis/fourier_silu_analysis.py
uv run python analysis/silu_ablation_comparison.py
uv run python analysis/eval_generalization.py
uv run python analysis/eval_generalization_checks.py
uv run python analysis/summarize_generalization.py
uv run python analysis/eval_unified_routing.py
uv run python analysis/routing_mi_silu_vs_gelu.py
uv run python analysis/reviewer_analyses.py

# Standalone figure scripts
uv run python figures/plot_silu_mechanisms.py
uv run python scripts/fig_silu_comparison.py
uv run python scripts/fig_narrow_ffn_modadd.py
uv run python scripts/plot_laux_sweep.py
```

See `analysis/README.md` for a per-script description of which figure each one produces and which checkpoints it consumes.

## Repository layout

```
model/         architecture (FFN/GLU/MoE/MoE-GLU + 1-layer transformer)
formerlens/    TransformerLens-compatible hooked variants for interpretability
data/          dataset classes for add-7, modular addition, histogram
train*.py      training entry points (add-7 = train.py, others named per-task)
scripts/       shell runners for multi-seed sweeps and reproducibility
analysis/      analysis + figure-generation scripts
figures/       generated figures (PNG / PDF / SVG) and standalone figure scripts
checkpoints/   trained model checkpoints (one directory per run)
archive/       superseded notebooks / scripts kept for reference
```

Subdirectory READMEs:

- [`model/README.md`](model/README.md) — architecture and component classes
- [`formerlens/README.md`](formerlens/README.md) — hook points and interpretability API
- [`data/README.md`](data/README.md) — dataset construction and tokenization
- [`scripts/README.md`](scripts/README.md) — per-script run counts and expected wall-clock times
- [`analysis/README.md`](analysis/README.md) — analysis scripts and figure mapping

## Checkpoint layout

After downloading or training, `checkpoints/` mirrors the run names used by the training scripts:

```
checkpoints/
  <task>_<arch>_<config>_s<seed>/
    best_model.pt          # add-7, histogram
    modadd_best.pt         # modular addition
```

`<task>` ∈ `{add7, modadd, hist}` · `<arch>` ∈ `{ffn, glu, moe, moe_glu}` · `<config>` encodes width / activation / normalization / routing variant (e.g. `nonorm`, `narrow_nonorm`, `topk2_nonorm`, `randroute_nonorm`, `d170_silu_nonorm`).

Each `.pt` file contains a Python dict with `model_state_dict`, `config`, and one of `accuracy` / `test_acc` / `step` / `epoch`. The `config` dict holds architectural hyperparameters only (no identifying metadata).

## Key tokens

- Digits: 0-9
- `PAD_TOKEN`: 10
- `EOS_TOKEN`: 11
- `REG_TOKEN`: 12 (register-token training only)

## License

See [LICENSE](LICENSE).
