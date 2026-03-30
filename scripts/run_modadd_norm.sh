#!/bin/bash
# Run all 4 variants x 5 seeds for modular addition WITH RMSNorm.
cd "$(dirname "$0")/.."
# Usage: bash run_modadd_norm.sh

SEEDS="42 137 256 512 1024"
FFN_TYPES="ffn glu moe moe_glu"
EPOCHS=40000

for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/modadd_${ftype}_norm_s${seed}"

        if [ -f "$dir/modadd_best.pt" ]; then
            echo "=== SKIP $ftype seed=$seed (already exists) ==="
            continue
        fi

        echo "=== Training $ftype seed=$seed (with norm) ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_modular_addition.py \
            --ffn_type "$ftype" \
            --seed "$seed" \
            --use_norm \
            --no_wandb \
            --epochs $EPOCHS \
            --checkpoint_dir "$dir" \
            --log_interval 500 \
            --save_interval 10000 \
            $extra_args

        echo "=== Done $ftype seed=$seed ==="
        echo
    done
done

echo "All runs complete!"
