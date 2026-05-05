#!/bin/bash
# Histogram Exp 3: Vary model width (d=48 for paper match, d=64, d=128 baseline, d=256)
cd "$(dirname "$0")/.."

SEEDS="42 137 256 512 1024"
EPOCHS=1000
WIDTHS="48 64 256"  # 128 is already done in baseline runs

echo "=== Histogram Exp 3: Vary Model Width ==="
echo

for width in $WIDTHS; do
    idim=$((width * 4))
    nheads=$((width / 32))
    if [ "$nheads" -lt 1 ]; then nheads=1; fi

    for ftype in ffn moe; do
        for seed in $SEEDS; do
            dir="checkpoints/hist_${ftype}_d${width}_s${seed}"
            if [ -f "$dir/hist_best.pt" ]; then echo "SKIP $ftype d=$width s$seed"; continue; fi
            echo "Training $ftype d_model=$width seed=$seed"

            extra_args=""
            if [ "$ftype" = "moe" ]; then
                extra_args="--num_experts 4"
            fi

            .venv/bin/python train_histogram.py \
                --ffn_type "$ftype" --model_dim "$width" --intermediate_dim "$idim" \
                --num_heads "$nheads" --seed "$seed" \
                --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir" \
                $extra_args
        done
    done
done

echo "Histogram Exp 3 complete!"
