#!/bin/bash
# Experiment 5: Top-k routing on modular addition
cd "$(dirname "$0")/.."
# top_k=1 (baseline, already done), top_k=2

SEEDS="42 137 256 512 1024"
EPOCHS=40000

echo "=== Experiment 5: Top-k Routing ==="
echo

# MoE top-2
for seed in $SEEDS; do
    dir="checkpoints/modadd_moe_topk2_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP MoE topk=2 s$seed"; continue; fi
    echo "Training MoE top_k=2 seed=$seed"
    .venv/bin/python train_modular_addition.py \
        --ffn_type moe --num_experts 4 --top_k 2 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir" \
        --log_interval 500 --save_interval 10000
done

# MoE-GLU top-2
for seed in $SEEDS; do
    dir="checkpoints/modadd_moe_glu_topk2_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP MoE-GLU topk=2 s$seed"; continue; fi
    echo "Training MoE-GLU top_k=2 seed=$seed"
    .venv/bin/python train_modular_addition.py \
        --ffn_type moe_glu --num_experts 4 --top_k 2 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir" \
        --log_interval 500 --save_interval 10000
done

echo "Experiment 5 complete!"
