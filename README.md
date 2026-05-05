# Sparsity Moves Computation: How FFN Architecture Reshapes Attention in Small Transformers

Code for studying how the choice of FFN architecture (dense FFN, GLU, MoE, MoE-GLU) shapes computation in 1-layer transformers on three algorithmic tasks: add-7, modular addition, and histogram counting.

## Installation

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Pre-trained checkpoints

All trained model checkpoints used in the paper (5 seeds per condition, every task / architecture / routing variant) are hosted anonymously on the Hugging Face Hub:

> <https://huggingface.co/Sparsity-Moves-Computation/moe-redistribution-checkpoints>

The repository can be downloaded without a Hugging Face account.

### Bulk download (recommended)

Pull the whole `checkpoints/` tree at once:

```bash
hf download Sparsity-Moves-Computation/moe-redistribution-checkpoints \
    --repo-type model \
    --local-dir checkpoints/
```

(Older `huggingface-cli` syntax: `huggingface-cli download Sparsity-Moves-Computation/moe-redistribution-checkpoints --local-dir checkpoints/`.)

The download is resumable; rerunning the same command skips files already present.

### Selective download (Python API)

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

### Layout

After downloading, the local `checkpoints/` directory mirrors the run names used by the training scripts:

```
checkpoints/
  <task>_<arch>_<config>_s<seed>/
    best_model.pt          # add-7, histogram
    modadd_best.pt         # modular addition
```

`<task>` ∈ `{add7, modadd, hist}` · `<arch>` ∈ `{ffn, glu, moe, moe_glu}` · `<config>` encodes width / activation / normalization / routing variant (e.g. `nonorm`, `narrow_nonorm`, `topk2_nonorm`, `randroute_nonorm`, `d170_silu_nonorm`).

Each `.pt` file contains a Python dict with `model_state_dict`, `config`, and one of `accuracy` / `test_acc` / `step` / `epoch`. The `config` dict holds architectural hyperparameters only (no identifying metadata).

## Quick start

Train one model:

```bash
uv run python train.py --ffn_type moe --num_digits 3 --no_wandb
```

Run a multi-seed sweep (these wrap `train.py` / `train_modular_addition.py` / `train_histogram.py`):

```bash
bash scripts/run_multiseed.sh                # modadd, 4 variants x 5 seeds
bash scripts/run_add7_multiseed.sh           # add-7, 4 variants x 5 seeds
bash scripts/run_histogram_multiseed.sh      # histogram, 4 variants x 5 seeds
```

Regenerate figures from trained checkpoints:

```bash
uv run python analysis/visualize_results.py             # main figures
uv run python analysis/make_matched_ablation_figures.py # parameter-matched ablations
uv run python analysis/make_matched_figures.py          # per-position + routing
uv run python analysis/make_matched_fourier_figure.py   # Fourier-concentration histograms
```

## Repository layout

```
model/         architecture (FFN/GLU/MoE/MoE-GLU + 1-layer transformer)
formerlens/    TransformerLens-compatible hooked variants for interpretability
data/          dataset classes for add-7, modular addition, histogram
train*.py      training entry points (add-7 = train.py, others named per-task)
scripts/       shell runners for multi-seed sweeps and reproducibility
analysis/      analysis + figure-generation scripts
figures/       generated figures (PNG / PDF / SVG)
checkpoints/   trained model checkpoints (one directory per run)
archive/       superseded notebooks / scripts kept for reference
```

Each subdirectory has its own README with details:

- [`model/README.md`](model/README.md) — architecture and component classes
- [`formerlens/README.md`](formerlens/README.md) — hook points and interpretability API
- [`data/README.md`](data/README.md) — dataset construction and tokenization
- [`scripts/README.md`](scripts/README.md) — full list of training scripts
- [`analysis/README.md`](analysis/README.md) — analysis scripts and figure mapping

## Key tokens

- Digits: 0-9
- `PAD_TOKEN`: 10
- `EOS_TOKEN`: 11
- `REG_TOKEN`: 12 (register-token training only)

## Reproducing a single result

The training scripts skip checkpoints that already exist, so a typical workflow is:

1. `bash scripts/<run_*.sh>` to train all seeds for a sweep, **or** download pre-trained checkpoints (see [Pre-trained checkpoints](#pre-trained-checkpoints)) to skip training entirely
2. `uv run python analysis/<figure_script>.py` to read the resulting checkpoints and write figures

Checkpoints are written to `checkpoints/<task>_<variant>_s<seed>/best_model.pt` (or `<task>_best.pt` for tasks that don't use a separate val split). The shell scripts pass the checkpoint directory as an argument, and the analysis scripts iterate seeds 42, 137, 256, 512, 1024.

## License

See [LICENSE](LICENSE).
