#!/bin/bash
# Per-active-matched MoE-GLU on add-7.
#
# Background: the codebase splits a *total* intermediate_dim across E experts
# (model/components.py: expert_intermediate = intermediate_dim // num_experts),
# so passing --intermediate_dim 170 yields h_E = 42 per expert (total-param
# matched to dense FFN, but per-active is only 1/4 of dense).
#
# To get a true per-active match (each expert's h_E = matched-GLU's h = 170),
# we pass intermediate_dim = E * h_glu = 4 * 170 = 680. This trains a model
# with h_E=170 per expert, ~4x dense total params, FLOP-matched per token.
#
# Output checkpoint directory uses "_d680_pa" suffix (pa = per-active) to
# distinguish from the existing total-matched "_d170_" runs.

cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
LOG="/tmp/per_active_match_add7.log"
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

log "=== Add-7 per-active-matched MoE-GLU (intermediate_dim=680, h_E=170) @ $(date) ==="

for seed in $SEEDS; do
    dir="checkpoints/add7_moe_glu_d680_pa_nonorm_s${seed}"
    if [ -f "$dir/best_model.pt" ]; then
        log "SKIP seed=$seed (existing checkpoint)"
        continue
    fi
    log "--- training seed=$seed @ $(date) ---"
    .venv/bin/python train.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 680 \
        --seed "$seed" \
        --num_digits 3 \
        --steps 10000 \
        --eval_interval 200 \
        --patience 50 \
        --no_use_norm \
        --checkpoint_dir "$dir" \
        --no_wandb 2>&1 | tee -a "$LOG"
    log "--- done seed=$seed @ $(date) ---"
done

log "=== ALL add-7 per-active-matched runs done @ $(date) ==="
