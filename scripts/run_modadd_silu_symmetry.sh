#!/bin/bash
# Modadd SiLU symmetry: fill the missing SiLU cells for the 4 main modadd variants.
# Existing modadd_{ffn,glu,moe,moe_glu}_s* were trained with default GELU.
# We now train matching variants with explicit --activation silu.
#
# Expected duration: ~30-45 min per seed x 5 seeds x 4 variants = ~10-15 hours CPU.
# Idempotent: skips any checkpoint dir that already has modadd_test99.pt OR best_model.pt.

cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
LOG="/tmp/modadd_silu_symmetry.log"
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

log "=== Modadd SiLU symmetry overnight @ $(date) ==="

train_modadd () {
    local ftype="$1" seed="$2"
    local dir="checkpoints/modadd_${ftype}_silu_s${seed}"
    # Skip if any grok marker OR completed-training marker OR valid best checkpoint exists.
    # train_modular_addition.py saves `modadd_best.pt` (best-so-far), `modadd_epoch40000.pt`
    # (final epoch), and `modadd_test99.pt` (99% grok marker). Older runs use `modadd_best.pt`.
    if [ -f "$dir/modadd_test99.pt" ] || [ -f "$dir/modadd_best.pt" ] || [ -f "$dir/best_model.pt" ] || [ -f "$dir/modadd_final.pt" ] || [ -f "$dir/modadd_epoch40000.pt" ]; then
        log "SKIP $dir (existing checkpoint found)"
        return
    fi
    log "--- modadd ${ftype} silu seed=${seed} @ $(date) ---"
    local extra=""
    if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
        extra="--num_experts 4 --top_k 1"
    fi
    .venv/bin/python train_modular_addition.py \
        --ffn_type "$ftype" \
        --activation silu \
        --seed "$seed" \
        --epochs 40000 \
        --checkpoint_dir "$dir" \
        --no_wandb \
        $extra 2>&1 | tail -5 | tee -a "$LOG"
    log "--- done ${ftype} silu seed=${seed} @ $(date) ---"
}

for ftype in ffn glu moe moe_glu; do
    log "=== ${ftype} SiLU seeds @ $(date) ==="
    for seed in $SEEDS; do
        train_modadd "$ftype" "$seed"
    done
    log "=== ${ftype} SiLU DONE @ $(date) ==="
done

log "=== ALL modadd SiLU symmetry DONE @ $(date) ==="
