#!/bin/bash
# Per-active-matched MoE-GLU on histogram.
#
# To get true per-active match (h_E = matched-GLU's h = 340), pass
# intermediate_dim = E * h_glu = 4 * 340 = 1360.

cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
LOG="/tmp/per_active_match_hist.log"
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

log "=== Histogram per-active-matched MoE-GLU (intermediate_dim=1360, h_E=340) @ $(date) ==="

for seed in $SEEDS; do
    dir="checkpoints/hist_moe_glu_d1360_pa_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then
        log "SKIP seed=$seed (existing checkpoint)"
        continue
    fi
    log "--- training seed=$seed @ $(date) ---"
    .venv/bin/python train_histogram.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 1360 \
        --seed "$seed" \
        --epochs 1000 \
        --checkpoint_dir "$dir" \
        --no_wandb 2>&1 | tee -a "$LOG"
    log "--- done seed=$seed @ $(date) ---"
done

log "=== ALL histogram per-active-matched runs done @ $(date) ==="
