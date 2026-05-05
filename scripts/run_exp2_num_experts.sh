#!/bin/bash
# Experiment 2: Vary number of experts on modular addition
cd "$(dirname "$0")/.."
# E=1 (FFN with useless router), E=2, E=4 (baseline), E=8, E=16

SEEDS="42 137 256 512 1024"
EPOCHS=40000
EXPERT_COUNTS="1 2 4 8 16"

echo "=== Experiment 2: Vary Number of Experts ==="
echo

for num_experts in $EXPERT_COUNTS; do
    for seed in $SEEDS; do
        dir="checkpoints/modadd_moe_e${num_experts}_s${seed}"
        if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP E=$num_experts s$seed"; continue; fi
        echo "Training MoE E=$num_experts seed=$seed"
        .venv/bin/python train_modular_addition.py \
            --ffn_type moe --num_experts "$num_experts" --seed "$seed" \
            --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir" \
            --log_interval 500 --save_interval 10000
    done
done

echo "Experiment 2 complete!"
