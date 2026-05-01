#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
# h=170 = (2/3)*256, total-parameter-matched GLU to dense FFN on add-7.
# Goal: ablation numbers under BOTH SiLU and GELU so the paper's activation claim can be verified either way.
#
# Existing state:
#   add7_glu_d170_gelu_nonorm_s*      GELU GLU h=170 (already run)
#   add7_moe_glu_d170_nonorm_s*       SiLU MoE-GLU h=170 (Phase 2, class-default SiLU)
# This script adds the missing pair:
#   add7_glu_d170_silu_nonorm_s*      SiLU GLU h=170
#   add7_moe_glu_d170_gelu_nonorm_s*  GELU MoE-GLU h=170

# ------------- Phase A: SiLU GLU h=170 -------------
echo "=== Phase A: add-7 GLU h=170 SiLU @ $(date) ==="
for seed in $SEEDS; do
    dir="checkpoints/add7_glu_d170_silu_nonorm_s${seed}"
    if [ -f "$dir/best_model.pt" ]; then
        echo "=== SKIP Phase A seed=$seed (exists) ==="
        continue
    fi
    echo "=== Train GLU SiLU h=170 seed=$seed @ $(date) ==="
    .venv/bin/python train.py \
        --ffn_type glu \
        --intermediate_dim 170 \
        --activation silu \
        --seed "$seed" \
        --num_digits 3 \
        --steps 10000 \
        --eval_interval 200 \
        --patience 50 \
        --no_use_norm \
        --checkpoint_dir "$dir" \
        --no_wandb
done
echo "=== Phase A DONE @ $(date) ==="

# ------------- Phase B: GELU MoE-GLU h=170 -------------
echo "=== Phase B: add-7 MoE-GLU h=170 GELU @ $(date) ==="
for seed in $SEEDS; do
    dir="checkpoints/add7_moe_glu_d170_gelu_nonorm_s${seed}"
    if [ -f "$dir/best_model.pt" ]; then
        echo "=== SKIP Phase B seed=$seed (exists) ==="
        continue
    fi
    echo "=== Train MoE-GLU GELU h=170 seed=$seed @ $(date) ==="
    .venv/bin/python train.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 170 \
        --activation gelu \
        --seed "$seed" \
        --num_digits 3 \
        --steps 10000 \
        --eval_interval 200 \
        --patience 50 \
        --no_use_norm \
        --checkpoint_dir "$dir" \
        --no_wandb
done
echo "=== Phase B DONE @ $(date) ==="
echo "=== ALL h=170 both-activation runs DONE @ $(date) ==="
