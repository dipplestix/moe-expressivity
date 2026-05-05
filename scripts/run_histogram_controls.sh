#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"

echo "=========================================="
echo "Random Routing MoE on histogram"
echo "=========================================="
for ftype in moe moe_glu; do
    for seed in $SEEDS; do
        dir="checkpoints/hist_${ftype}_randroute_s${seed}"
        if [ -f "$dir/hist_best.pt" ]; then
            echo "=== SKIP hist $ftype randroute seed=$seed ==="
            continue
        fi
        echo "=== Training hist $ftype randroute seed=$seed ==="
        .venv/bin/python train_histogram.py \
            --ffn_type "$ftype" \
            --num_experts 4 \
            --top_k 1 \
            --random_routing \
            --seed "$seed" \
            --epochs 1000 \
            --checkpoint_dir "$dir" \
            --no_wandb
        echo "=== Done hist $ftype randroute seed=$seed ==="
    done
done

echo "=========================================="
echo "Narrow FFN on histogram (intermediate_dim 128)"
echo "=========================================="
# Histogram default intermediate_dim is 512, narrow = 128 (matches per-token MoE capacity)
for seed in $SEEDS; do
    dir="checkpoints/hist_ffn_narrow_s${seed}"
    if [ -f "$dir/hist_best.pt" ]; then
        echo "=== SKIP hist ffn_narrow seed=$seed ==="
        continue
    fi
    echo "=== Training hist ffn_narrow (idim=128) seed=$seed ==="
    .venv/bin/python train_histogram.py \
        --ffn_type ffn \
        --intermediate_dim 128 \
        --seed "$seed" \
        --epochs 1000 \
        --checkpoint_dir "$dir" \
        --no_wandb
    echo "=== Done hist ffn_narrow seed=$seed ==="
done

echo "All histogram control experiments complete!"
