#!/bin/bash
# SiLU per-active-matched MoE-GLU on all three tasks, chained.
# Detached version (run under nohup) so it survives window close.
#
# Each run uses --activation silu and the same per-active intermediate_dim
# as the GELU per-active runs (680 add-7, 1360 modadd/hist).

cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
LOG_ROOT="/tmp/per_active_match_silu"
: > "${LOG_ROOT}_chain.log"

log() { echo "$@" | tee -a "${LOG_ROOT}_chain.log"; }

# ---------------- add-7 ----------------
log "=== add-7 SiLU per-active-matched MoE-GLU @ $(date) ==="
for seed in $SEEDS; do
    dir="checkpoints/add7_moe_glu_d680_silu_pa_nonorm_s${seed}"
    if [ -f "$dir/best_model.pt" ]; then
        log "SKIP add-7 seed=$seed"
        continue
    fi
    log "--- add-7 silu seed=$seed @ $(date) ---"
    .venv/bin/python train.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 680 \
        --activation silu \
        --seed "$seed" \
        --num_digits 3 \
        --steps 10000 \
        --eval_interval 200 \
        --patience 50 \
        --no_use_norm \
        --checkpoint_dir "$dir" \
        --no_wandb 2>&1 | tee -a "${LOG_ROOT}_add7.log"
done
log "=== add-7 SiLU done @ $(date) ==="

# ---------------- histogram ----------------
log "=== hist SiLU per-active-matched MoE-GLU @ $(date) ==="
for seed in $SEEDS; do
    dir="checkpoints/hist_moe_glu_d1360_silu_pa_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then
        log "SKIP hist seed=$seed"
        continue
    fi
    log "--- hist silu seed=$seed @ $(date) ---"
    .venv/bin/python train_histogram.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 1360 \
        --activation silu \
        --seed "$seed" \
        --epochs 1000 \
        --checkpoint_dir "$dir" \
        --no_wandb 2>&1 | tee -a "${LOG_ROOT}_hist.log"
done
log "=== hist SiLU done @ $(date) ==="

# ---------------- modadd ----------------
log "=== modadd SiLU per-active-matched MoE-GLU @ $(date) ==="
for seed in $SEEDS; do
    dir="checkpoints/modadd_moe_glu_d1360_silu_pa_s${seed}"
    if [ -f "$dir/modadd_best.pt" ] || [ -f "$dir/modadd_test99.pt" ] || [ -f "$dir/modadd_epoch40000.pt" ]; then
        log "SKIP modadd seed=$seed"
        continue
    fi
    log "--- modadd silu seed=$seed @ $(date) ---"
    .venv/bin/python train_modular_addition.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 1360 \
        --activation silu \
        --seed "$seed" \
        --epochs 40000 \
        --checkpoint_dir "$dir" \
        --no_wandb 2>&1 | tee -a "${LOG_ROOT}_modadd.log"
done
log "=== modadd SiLU done @ $(date) ==="

log "=== ALL SiLU per-active-matched runs DONE @ $(date) ==="
