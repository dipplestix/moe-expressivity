# scripts/

Shell scripts for running all experiments. Each script is self-contained and can be run from any directory (it `cd`s to the project root automatically). Scripts skip runs that already have checkpoints.

## How to Run

```bash
bash scripts/<script_name>.sh
```

## Modular Addition Experiments

| Script | What it tests | # Runs | Approx. time (CPU) |
|--------|--------------|--------|---------------------|
| `run_multiseed.sh` | Core 4 variants (FFN/GLU/MoE/MoE-GLU) x 5 seeds, 40k epochs | 20 | ~8 hours |
| `run_exp1_regularization_baselines.sh` | FFN + dropout(0.1, 0.3) + weight decay(0.5, 2.0) x 5 seeds | 20 | ~8 hours |
| `run_exp2_num_experts.sh` | MoE with E=1,2,4,8,16 x 5 seeds | 25 | ~10 hours |
| `run_exp3_model_width.sh` | d_model=64,256 for FFN and MoE x 5 seeds | 20 | ~10 hours |
| `run_exp5_top_k.sh` | MoE and MoE-GLU with top_k=2 x 5 seeds | 10 | ~4 hours |
| `run_modadd_norm.sh` | All 4 variants with RMSNorm x 5 seeds | 20 | ~8 hours |

## Add-7 Experiments

| Script | What it tests | # Runs | Approx. time (CPU) |
|--------|--------------|--------|---------------------|
| `run_add7_multiseed.sh` | Core 4 variants x 5 seeds, 10k steps | 20 | ~2 hours |
| `run_add7_nonorm.sh` | All 4 variants without RMSNorm x 5 seeds | 20 | ~2 hours |

## Histogram Experiments

| Script | What it tests | # Runs | Approx. time (CPU) |
|--------|--------------|--------|---------------------|
| `run_histogram_multiseed.sh` | Core 4 variants x 5 seeds, 500 epochs | 20 | ~4 hours |
| `run_hist_exp1_regularization.sh` | FFN + dropout(0.1, 0.3) + weight decay(0.5, 2.0) x 5 seeds | 20 | ~4 hours |
| `run_hist_exp2_num_experts.sh` | MoE with E=1,2,4,8,16 x 5 seeds | 25 | ~5 hours |
| `run_hist_exp3_width.sh` | d=48,64,256 for FFN and MoE x 5 seeds | 30 | ~6 hours |
| `run_hist_exp5_topk.sh` | MoE and MoE-GLU with top_k=2 x 5 seeds | 10 | ~2 hours |

## Checkpoint Naming Convention

Checkpoints are saved to `checkpoints/<task>_<variant>_<modifier>_s<seed>/`:

```
checkpoints/
  modadd_ffn_s42/              # Modular addition, FFN, seed 42
  modadd_moe_e8_s137/          # Modular addition, MoE with 8 experts, seed 137
  modadd_ffn_drop01_s42/       # Modular addition, FFN + dropout 0.1, seed 42
  modadd_ffn_d256_s42/         # Modular addition, FFN, d_model=256, seed 42
  modadd_moe_topk2_s42/        # Modular addition, MoE top-2, seed 42
  modadd_ffn_norm_s42/         # Modular addition, FFN with RMSNorm, seed 42
  add7_ffn_s42/                # Add-7, FFN, seed 42 (with norm, default)
  add7_ffn_nonorm_s42/         # Add-7, FFN, no norm, seed 42
  hist_ffn_s42/                # Histogram, FFN, seed 42
  hist_moe_e8_s42/             # Histogram, MoE with 8 experts, seed 42
```

Each checkpoint directory contains:
- `*_best.pt` — Best test accuracy model
- `*_epoch{N}.pt` — Periodic saves
- `*_test{50,90,99}.pt` — Milestone checkpoints (when test acc crosses thresholds)

## Completion Status

### Done:
- All modular addition experiments (core + all controlled experiments)
- All add-7 experiments (core + norm/no-norm)
- Histogram core training (4 variants x 5 seeds, 1000 epochs)

### Not yet run:
- Histogram controlled experiments (regularization, num experts, width, top-k)

## Recommended Run Order

1. Core multi-seed runs (all 3 tasks) — **done**
2. Regularization baselines (exp 1) — **done** (modadd), pending (histogram)
3. Number of experts (exp 2) — **done** (modadd), pending (histogram)
4. Top-k routing (exp 5) — **done** (modadd), pending (histogram)
5. Model width (exp 3) — **done** (modadd), pending (histogram)
6. Norm ablations — **done** (both tasks)
