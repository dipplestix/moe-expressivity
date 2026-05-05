#!/bin/bash
# Modadd d=340 SiLU controls: fill the missing SiLU pair for the param-matched
# GLU and MoE-GLU controls on modular addition. GELU runs already exist at
# modadd_{glu,moe_glu}_d340_s*; this script trains explicit SiLU pairs.
#
# Expected duration: ~30-45 min per seed x 5 seeds x 2 variants = ~5-7 hours CPU.
# Idempotent: skips any checkpoint dir that already has a valid checkpoint.

cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
LOG="/tmp/modadd_d340_silu.log"
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

log "=== Modadd d=340 SiLU @ $(date) ==="

train_modadd () {
    local ftype="$1" seed="$2"
    local dir="checkpoints/modadd_${ftype}_d340_silu_s${seed}"
    if [ -f "$dir/modadd_test99.pt" ] || [ -f "$dir/modadd_best.pt" ] || [ -f "$dir/best_model.pt" ] || [ -f "$dir/modadd_final.pt" ] || [ -f "$dir/modadd_epoch40000.pt" ]; then
        log "SKIP $dir (existing checkpoint found)"
        return
    fi
    log "--- modadd ${ftype} d=340 silu seed=${seed} @ $(date) ---"
    local extra=""
    if [ "$ftype" = "moe_glu" ]; then
        extra="--num_experts 4 --top_k 1"
    fi
    .venv/bin/python train_modular_addition.py \
        --ffn_type "$ftype" \
        --activation silu \
        --intermediate_dim 340 \
        --seed "$seed" \
        --epochs 40000 \
        --checkpoint_dir "$dir" \
        --no_wandb \
        $extra 2>&1 | tail -5 | tee -a "$LOG"
    log "--- done ${ftype} d=340 silu seed=${seed} @ $(date) ---"
}

for ftype in glu moe_glu; do
    log "=== ${ftype} d=340 SiLU seeds @ $(date) ==="
    for seed in $SEEDS; do
        train_modadd "$ftype" "$seed"
    done
    log "=== ${ftype} d=340 SiLU DONE @ $(date) ==="
done

log "=== ALL modadd d=340 SiLU DONE @ $(date) ==="
