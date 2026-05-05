#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
# Add-7: model_dim=64, dense FFN intermediate_dim=256, dense FFN params = 2*64*256 = 32768.
# Param-matched (per-active) MoE-GLU: h = 32768/(3*64) = 170.67 -> 170.
echo "=== Param-matched MoE-GLU (intermediate_dim=170) on add-7 ==="
for seed in $SEEDS; do
    dir="checkpoints/add7_moe_glu_d170_nonorm_s${seed}"
    if [ -f "$dir/best_model.pt" ]; then
        echo "=== SKIP seed=$seed ==="
        continue
    fi
    echo "=== Training add7 moe_glu d=170 seed=$seed ==="
    .venv/bin/python train.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 170 \
        --seed "$seed" \
        --num_digits 3 \
        --steps 10000 \
        --eval_interval 200 \
        --patience 50 \
        --no_use_norm \
        --checkpoint_dir "$dir" \
        --no_wandb
    echo "=== Done seed=$seed ==="
done
echo "All add-7 param-matched MoE-GLU done!"
