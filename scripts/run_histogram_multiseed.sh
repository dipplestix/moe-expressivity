#!/bin/bash
# Run all 4 variants x 5 seeds for histogram counting task.
# Usage: bash scripts/run_histogram_multiseed.sh
cd "$(dirname "$0")/.."

SEEDS="42 137 256 512 1024"
FFN_TYPES="ffn glu moe moe_glu"
EPOCHS=1000

for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/hist_${ftype}_s${seed}"

        if [ -f "$dir/hist_best.pt" ]; then
            echo "=== SKIP $ftype seed=$seed (already exists) ==="
            continue
        fi

        echo "=== Training $ftype seed=$seed ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_histogram.py \
            --ffn_type "$ftype" \
            --seed "$seed" \
            --no_wandb \
            --epochs $EPOCHS \
            --checkpoint_dir "$dir" \
            $extra_args

        echo "=== Done $ftype seed=$seed ==="
        echo
    done
done

echo "All histogram runs complete!"
