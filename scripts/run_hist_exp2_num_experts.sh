#!/bin/bash
# Histogram Exp 2: Vary number of experts (E=1,2,4,8,16)
cd "$(dirname "$0")/.."

SEEDS="42 137 256 512 1024"
EPOCHS=500
EXPERT_COUNTS="1 2 4 8 16"

echo "=== Histogram Exp 2: Vary Number of Experts ==="
echo

for num_experts in $EXPERT_COUNTS; do
    for seed in $SEEDS; do
        dir="checkpoints/hist_moe_e${num_experts}_s${seed}"
        if [ -f "$dir/hist_best.pt" ]; then echo "SKIP E=$num_experts s$seed"; continue; fi
        echo "Training MoE E=$num_experts seed=$seed"
        .venv/bin/python train_histogram.py \
            --ffn_type moe --num_experts "$num_experts" --seed "$seed" \
            --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir"
    done
done

echo "Histogram Exp 2 complete!"
