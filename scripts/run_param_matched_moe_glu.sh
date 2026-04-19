#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
# Modadd: dense FFN 131k params. MoE-GLU intermediate_dim=340 matches (131k, 0.5% off)
echo "=== Param-matched MoE-GLU (intermediate_dim=340) on modadd ==="
for seed in $SEEDS; do
    dir="checkpoints/modadd_moe_glu_d340_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then
        echo "=== SKIP seed=$seed ==="
        continue
    fi
    echo "=== Training modadd moe_glu d=340 seed=$seed ==="
    .venv/bin/python train_modular_addition.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 340 \
        --seed "$seed" \
        --epochs 40000 \
        --checkpoint_dir "$dir" \
        --no_wandb
    echo "=== Done seed=$seed ==="
done
echo "All param-matched MoE-GLU done!"
