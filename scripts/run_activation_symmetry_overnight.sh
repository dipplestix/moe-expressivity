#!/bin/bash
# Overnight activation-symmetry runs.
#
# Current activation state (confirmed by spot-checks on existing checkpoints):
#   add-7  main:        activation=None  -> SiLU (class default, NOT GELU as paper line 226 claims)
#   add-7  h=170 GLU:   SiLU + GELU (done)
#   add-7  h=170 MoE-GLU: SiLU (Phase 2) + GELU (Phase B, in progress)
#   histogram all:       activation='gelu' -> GELU explicit
#   modadd all:          activation='gelu' -> GELU explicit
#
# This script fills the missing activation cells we CAN run overnight:
#   Tier 1: add-7 main variants with explicit GELU  (4 variants x 5 seeds = 20 runs)
#   Tier 2: histogram main variants with explicit SiLU (4 variants x 5 seeds = 20 runs)
#   Tier 3: histogram h=340 param-matched with SiLU (2 variants x 5 seeds = 10 runs)
#
# Modadd SiLU (40k epochs * 20 runs ~ 15 hrs) is out of scope here -- flagged for morning.

cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
LOG="/tmp/activation_symmetry.log"
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

log "=== Activation Symmetry Overnight @ $(date) ==="

# ---------- helpers ----------
train_add7 () {
    local ftype="$1" activation="$2" seed="$3" suffix="$4"
    local dir="checkpoints/add7_${ftype}_${activation}_nonorm_s${seed}"
    if [ -f "$dir/best_model.pt" ]; then
        log "SKIP $dir"
        return
    fi
    log "--- add7 ${ftype} ${activation} seed=${seed} @ $(date) ---"
    local extra=""
    if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
        extra="--num_experts 4 --top_k 1"
    fi
    .venv/bin/python train.py \
        --ffn_type "$ftype" \
        --activation "$activation" \
        --seed "$seed" \
        --num_digits 3 --steps 10000 --eval_interval 200 --patience 50 \
        --no_use_norm \
        --checkpoint_dir "$dir" \
        --no_wandb \
        $extra 2>&1 | tail -5 | tee -a "$LOG"
}

train_hist () {
    local ftype="$1" activation="$2" seed="$3" idim="$4"
    local tag="${activation}"
    local dir
    if [ -n "$idim" ]; then
        dir="checkpoints/hist_${ftype}_d${idim}_${tag}_s${seed}"
    else
        dir="checkpoints/hist_${ftype}_${tag}_s${seed}"
    fi
    if [ -f "$dir/hist_best.pt" ] || [ -f "$dir/best_model.pt" ]; then
        log "SKIP $dir"
        return
    fi
    log "--- hist ${ftype} ${activation} ${idim:+d=$idim} seed=${seed} @ $(date) ---"
    local extra=""
    if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
        extra="--num_experts 4 --top_k 1"
    fi
    local idim_arg=""
    if [ -n "$idim" ]; then
        idim_arg="--intermediate_dim $idim"
    fi
    .venv/bin/python train_histogram.py \
        --ffn_type "$ftype" \
        --activation "$activation" \
        --seed "$seed" \
        --epochs 1000 \
        --checkpoint_dir "$dir" \
        --no_wandb \
        $idim_arg $extra 2>&1 | tail -5 | tee -a "$LOG"
}

# ---------- Tier 1: add-7 GELU ----------
log "=== Tier 1: add-7 main variants GELU @ $(date) ==="
for ftype in ffn glu moe moe_glu; do
    for seed in $SEEDS; do
        train_add7 "$ftype" "gelu" "$seed"
    done
done
log "=== Tier 1 DONE @ $(date) ==="

# ---------- Tier 2: histogram SiLU (default idim) ----------
log "=== Tier 2: histogram main variants SiLU @ $(date) ==="
for ftype in ffn glu moe moe_glu; do
    for seed in $SEEDS; do
        train_hist "$ftype" "silu" "$seed" ""
    done
done
log "=== Tier 2 DONE @ $(date) ==="

# ---------- Tier 3: histogram SiLU param-matched ----------
log "=== Tier 3: histogram param-matched (d=340) SiLU @ $(date) ==="
for ftype in glu moe_glu; do
    for seed in $SEEDS; do
        train_hist "$ftype" "silu" "$seed" "340"
    done
done
log "=== Tier 3 DONE @ $(date) ==="

log "=== ALL activation-symmetry runs DONE @ $(date) ==="
