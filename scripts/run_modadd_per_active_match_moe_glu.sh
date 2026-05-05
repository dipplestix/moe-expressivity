#!/bin/bash
# Per-active-matched MoE-GLU on modular addition.
#
# To get true per-active match (h_E = matched-GLU's h = 340), pass
# intermediate_dim = E * h_glu = 4 * 340 = 1360. This trains a model with
# h_E=340 per expert, ~4x dense total params, FLOP-matched per token.

cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
LOG="/tmp/per_active_match_modadd.log"
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

log "=== Modadd per-active-matched MoE-GLU (intermediate_dim=1360, h_E=340) @ $(date) ==="

for seed in $SEEDS; do
    dir="checkpoints/modadd_moe_glu_d1360_pa_s${seed}"
    if [ -f "$dir/modadd_best.pt" ] || [ -f "$dir/modadd_test99.pt" ] || [ -f "$dir/modadd_epoch40000.pt" ]; then
        log "SKIP seed=$seed (existing checkpoint)"
        continue
    fi
    log "--- training seed=$seed @ $(date) ---"
    .venv/bin/python train_modular_addition.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 1360 \
        --seed "$seed" \
        --epochs 40000 \
        --checkpoint_dir "$dir" \
        --no_wandb 2>&1 | tee -a "$LOG"
    log "--- done seed=$seed @ $(date) ---"
done

log "=== ALL modadd per-active-matched runs done @ $(date) ==="
