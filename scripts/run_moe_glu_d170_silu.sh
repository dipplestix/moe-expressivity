#!/bin/bash
# Fills the missing cell: add-7 MoE-GLU h=170 with EXPLICIT SiLU.
# (Phase 2 `add7_moe_glu_d170_nonorm_s*` is actually GELU — train.py default.)
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
LOG="/tmp/moe_glu_d170_silu.log"
: > "$LOG"
echo "=== add-7 MoE-GLU h=170 SiLU (explicit) @ $(date) ===" | tee -a "$LOG"
for seed in $SEEDS; do
    dir="checkpoints/add7_moe_glu_d170_silu_nonorm_s${seed}"
    if [ -f "$dir/best_model.pt" ]; then
        echo "SKIP $dir" | tee -a "$LOG"
        continue
    fi
    echo "--- seed=$seed @ $(date) ---" | tee -a "$LOG"
    .venv/bin/python train.py \
        --ffn_type moe_glu \
        --num_experts 4 --top_k 1 \
        --intermediate_dim 170 \
        --activation silu \
        --seed "$seed" \
        --num_digits 3 --steps 10000 --eval_interval 200 --patience 50 \
        --no_use_norm \
        --checkpoint_dir "$dir" \
        --no_wandb 2>&1 | tail -5 | tee -a "$LOG"
done
echo "=== MoE-GLU h=170 SiLU DONE @ $(date) ===" | tee -a "$LOG"
