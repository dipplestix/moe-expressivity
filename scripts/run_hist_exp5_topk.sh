#!/bin/bash
# Histogram Exp 5: Top-k routing (top-1 baseline already done, test top-2)
cd "$(dirname "$0")/.."

SEEDS="42 137 256 512 1024"
EPOCHS=500

echo "=== Histogram Exp 5: Top-k Routing ==="
echo

# MoE top-2
for seed in $SEEDS; do
    dir="checkpoints/hist_moe_topk2_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then echo "SKIP MoE topk=2 s$seed"; continue; fi
    echo "Training MoE top_k=2 seed=$seed"
    .venv/bin/python train_histogram.py \
        --ffn_type moe --num_experts 4 --top_k 2 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir"
done

# MoE-GLU top-2
for seed in $SEEDS; do
    dir="checkpoints/hist_moe_glu_topk2_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then echo "SKIP MoE-GLU topk=2 s$seed"; continue; fi
    echo "Training MoE-GLU top_k=2 seed=$seed"
    .venv/bin/python train_histogram.py \
        --ffn_type moe_glu --num_experts 4 --top_k 2 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir"
done

echo "Histogram Exp 5 complete!"
