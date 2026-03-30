#!/bin/bash
# Experiment 1: Regularization baselines on modular addition
cd "$(dirname "$0")/.."
# Tests whether MoE's grokking speedup is from routing or just extra regularization.
# Compares: FFN+dropout(0.1), FFN+dropout(0.3), FFN+heavier weight decay
# All use seed 42 only first to check, then multi-seed if promising.

SEEDS="42 137 256 512 1024"
EPOCHS=40000

echo "=== Experiment 1: Regularization Baselines ==="
echo

# Baseline 1: FFN + dropout 0.1
for seed in $SEEDS; do
    dir="checkpoints/modadd_ffn_drop01_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP ffn+drop0.1 s$seed"; continue; fi
    echo "Training FFN+dropout=0.1 seed=$seed"
    .venv/bin/python train_modular_addition.py \
        --ffn_type ffn --dropout 0.1 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir" \
        --log_interval 500 --save_interval 10000
done

# Baseline 2: FFN + dropout 0.3
for seed in $SEEDS; do
    dir="checkpoints/modadd_ffn_drop03_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP ffn+drop0.3 s$seed"; continue; fi
    echo "Training FFN+dropout=0.3 seed=$seed"
    .venv/bin/python train_modular_addition.py \
        --ffn_type ffn --dropout 0.3 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir" \
        --log_interval 500 --save_interval 10000
done

# Baseline 3: FFN + heavier weight decay (2.0 instead of 1.0)
for seed in $SEEDS; do
    dir="checkpoints/modadd_ffn_wd2_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP ffn+wd2 s$seed"; continue; fi
    echo "Training FFN+weight_decay=2.0 seed=$seed"
    .venv/bin/python train_modular_addition.py \
        --ffn_type ffn --weight_decay 2.0 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir" \
        --log_interval 500 --save_interval 10000
done

# Baseline 4: FFN + lighter weight decay (0.5)
for seed in $SEEDS; do
    dir="checkpoints/modadd_ffn_wd05_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP ffn+wd0.5 s$seed"; continue; fi
    echo "Training FFN+weight_decay=0.5 seed=$seed"
    .venv/bin/python train_modular_addition.py \
        --ffn_type ffn --weight_decay 0.5 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir" \
        --log_interval 500 --save_interval 10000
done

echo "Experiment 1 complete!"
