#!/bin/bash
# Run all 4 variants x 5 seeds for add-7 WITHOUT RMSNorm.
cd "$(dirname "$0")/.."
# Usage: bash run_add7_nonorm.sh

SEEDS="42 137 256 512 1024"
FFN_TYPES="ffn glu moe moe_glu"
STEPS=10000
NUM_DIGITS=3

for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/add7_${ftype}_nonorm_s${seed}"

        if [ -f "$dir/best_model.pt" ]; then
            echo "=== SKIP $ftype seed=$seed (already exists) ==="
            continue
        fi

        echo "=== Training $ftype seed=$seed (no norm) ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train.py \
            --ffn_type "$ftype" \
            --seed "$seed" \
            --num_digits $NUM_DIGITS \
            --steps $STEPS \
            --eval_interval 200 \
            --patience 50 \
            --no_wandb \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            $extra_args

        echo "=== Done $ftype seed=$seed ==="
        echo
    done
done

echo "All runs complete!"
