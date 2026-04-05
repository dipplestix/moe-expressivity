#!/bin/bash
# Histogram Exp 1: Regularization baselines
# FFN + dropout(0.1, 0.3), FFN + weight decay(0.5, 2.0)
cd "$(dirname "$0")/.."

SEEDS="42 137 256 512 1024"
EPOCHS=1000

echo "=== Histogram Exp 1: Regularization Baselines ==="
echo

# FFN + dropout 0.1
for seed in $SEEDS; do
    dir="checkpoints/hist_ffn_drop01_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then echo "SKIP ffn+drop0.1 s$seed"; continue; fi
    echo "Training FFN+dropout=0.1 seed=$seed"
    .venv/bin/python train_histogram.py \
        --ffn_type ffn --dropout 0.1 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir"
done

# FFN + dropout 0.3
for seed in $SEEDS; do
    dir="checkpoints/hist_ffn_drop03_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then echo "SKIP ffn+drop0.3 s$seed"; continue; fi
    echo "Training FFN+dropout=0.3 seed=$seed"
    .venv/bin/python train_histogram.py \
        --ffn_type ffn --dropout 0.3 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir"
done

# FFN + weight decay 2.0
for seed in $SEEDS; do
    dir="checkpoints/hist_ffn_wd2_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then echo "SKIP ffn+wd2 s$seed"; continue; fi
    echo "Training FFN+weight_decay=2.0 seed=$seed"
    .venv/bin/python train_histogram.py \
        --ffn_type ffn --weight_decay 2.0 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir"
done

# FFN + weight decay 0.5
for seed in $SEEDS; do
    dir="checkpoints/hist_ffn_wd05_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then echo "SKIP ffn+wd0.5 s$seed"; continue; fi
    echo "Training FFN+weight_decay=0.5 seed=$seed"
    .venv/bin/python train_histogram.py \
        --ffn_type ffn --weight_decay 0.5 --seed "$seed" \
        --no_wandb --epochs $EPOCHS --checkpoint_dir "$dir"
done

echo "Histogram Exp 1 complete!"
