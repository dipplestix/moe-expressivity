#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
# Histogram: model_dim=128, dense FFN intermediate_dim=512, dense FFN params = 2*128*512 = 131k.
# Param-matched GLU: h = 131k/(3*128) = 340.
# Param-matched (per-active) MoE-GLU: same h = 340.

echo "=========================================="
echo "Param-matched GLU (intermediate_dim=340) on histogram"
echo "=========================================="
for seed in $SEEDS; do
    dir="checkpoints/hist_glu_d340_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then
        echo "=== SKIP hist glu d=340 seed=$seed ==="
        continue
    fi
    echo "=== Training hist glu d=340 seed=$seed ==="
    .venv/bin/python train_histogram.py \
        --ffn_type glu \
        --intermediate_dim 340 \
        --seed "$seed" \
        --epochs 1000 \
        --checkpoint_dir "$dir" \
        --no_wandb
    echo "=== Done hist glu d=340 seed=$seed ==="
done

echo "=========================================="
echo "Param-matched MoE-GLU (intermediate_dim=340) on histogram"
echo "=========================================="
for seed in $SEEDS; do
    dir="checkpoints/hist_moe_glu_d340_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then
        echo "=== SKIP hist moe_glu d=340 seed=$seed ==="
        continue
    fi
    echo "=== Training hist moe_glu d=340 seed=$seed ==="
    .venv/bin/python train_histogram.py \
        --ffn_type moe_glu \
        --num_experts 4 \
        --top_k 1 \
        --intermediate_dim 340 \
        --seed "$seed" \
        --epochs 1000 \
        --checkpoint_dir "$dir" \
        --no_wandb
    echo "=== Done hist moe_glu d=340 seed=$seed ==="
done

echo "All histogram param-matched experiments complete!"
