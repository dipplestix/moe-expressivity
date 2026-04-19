#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"

for ftype in ffn glu moe moe_glu; do
    for freeze in attention ffn; do
        for seed in $SEEDS; do
            dir="checkpoints/hist_${ftype}_frozen${freeze}_s${seed}"
            if [ -f "$dir/hist_best.pt" ]; then
                echo "=== SKIP hist $ftype frozen-$freeze seed=$seed ==="
                continue
            fi
            echo "=== Training hist $ftype frozen-$freeze seed=$seed ==="
            extra_args=""
            if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
                extra_args="--num_experts 4"
            fi
            .venv/bin/python train_frozen_histogram.py \
                --ffn_type "$ftype" \
                --freeze "$freeze" \
                --seed "$seed" \
                --epochs 1000 \
                --patience 50 \
                --checkpoint_dir "$dir" \
                --no_wandb \
                $extra_args
            echo "=== Done hist $ftype frozen-$freeze seed=$seed ==="
            echo
        done
    done
done
echo "All frozen histogram experiments complete!"
