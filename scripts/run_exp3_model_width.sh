#!/bin/bash
# Experiment 3: Vary model width on modular addition
cd "$(dirname "$0")/.."
# Keep ffn_dim = 4 * model_dim (default ratio)
# d_model=64 (smaller), d_model=128 (baseline), d_model=256 (larger)

SEEDS="42 137 256 512 1024"
EPOCHS=40000
WIDTHS="64 256"  # 128 is already done in baseline runs

echo "=== Experiment 3: Vary Model Width ==="
echo

for width in $WIDTHS; do
    idim=$((width * 4))
    nheads=$((width / 32))  # keep head_dim=32
    if [ "$nheads" -lt 1 ]; then nheads=1; fi

    for ftype in ffn moe; do
        for seed in $SEEDS; do
            dir="checkpoints/modadd_${ftype}_d${width}_s${seed}"
            if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP $ftype d=$width s$seed"; continue; fi
            echo "Training $ftype d_model=$width seed=$seed"

            extra_args=""
            if [ "$ftype" = "moe" ]; then
                extra_args="--num_experts 4"
            fi

            .venv/bin/python train_modular_addition.py \
                --ffn_type "$ftype" --model_dim "$width" --intermediate_dim "$idim" \
                --num_heads "$nheads" --seed "$seed" \
                --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir" \
                --log_interval 500 --save_interval 10000 \
                $extra_args
        done
    done
done

echo "Experiment 3 complete!"
