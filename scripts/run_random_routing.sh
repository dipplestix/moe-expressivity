#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"

echo "=========================================="
echo "Random Routing MoE on modadd"
echo "=========================================="
for ftype in moe moe_glu; do
    for seed in $SEEDS; do
        dir="checkpoints/modadd_${ftype}_randroute_s${seed}"
        if [ -f "$dir/modadd_best.pt" ]; then
            echo "=== SKIP modadd $ftype randroute seed=$seed ==="
            continue
        fi
        echo "=== Training modadd $ftype randroute seed=$seed ==="
        .venv/bin/python train_modular_addition.py \
            --ffn_type "$ftype" \
            --num_experts 4 \
            --top_k 1 \
            --random_routing \
            --seed "$seed" \
            --epochs 40000 \
            --checkpoint_dir "$dir" \
            --no_wandb
        echo "=== Done modadd $ftype randroute seed=$seed ==="
    done
done

echo "=========================================="
echo "Random Routing MoE on add-7"
echo "=========================================="
for ftype in moe moe_glu; do
    for seed in $SEEDS; do
        dir="checkpoints/add7_${ftype}_randroute_nonorm_s${seed}"
        if [ -f "$dir/best_model.pt" ]; then
            echo "=== SKIP add7 $ftype randroute seed=$seed ==="
            continue
        fi
        echo "=== Training add7 $ftype randroute seed=$seed ==="
        .venv/bin/python train.py \
            --ffn_type "$ftype" \
            --num_experts 4 \
            --top_k 1 \
            --random_routing \
            --seed "$seed" \
            --num_digits 3 \
            --steps 10000 \
            --eval_interval 200 \
            --patience 50 \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            --no_wandb
        echo "=== Done add7 $ftype randroute seed=$seed ==="
    done
done

echo "All random routing experiments complete!"
