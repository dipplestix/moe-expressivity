#!/bin/bash
cd "$(dirname "$0")/.."
# Original grokking used seeds 42, 137, 256, 512, 1024. Add 15 more.
EXTRA_SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

echo "=== Extra grokking seeds: dense FFN on modadd ==="
for seed in $EXTRA_SEEDS; do
    dir="checkpoints/modadd_ffn_extra_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then
        echo "=== SKIP ffn seed=$seed ==="
        continue
    fi
    echo "=== Training modadd ffn seed=$seed ==="
    .venv/bin/python train_modular_addition.py \
        --ffn_type ffn \
        --seed "$seed" \
        --epochs 40000 \
        --checkpoint_dir "$dir" \
        --no_wandb
    echo "=== Done ffn seed=$seed ==="
done

echo "=== Extra grokking seeds: MoE on modadd ==="
for seed in $EXTRA_SEEDS; do
    dir="checkpoints/modadd_moe_extra_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then
        echo "=== SKIP moe seed=$seed ==="
        continue
    fi
    echo "=== Training modadd moe seed=$seed ==="
    .venv/bin/python train_modular_addition.py \
        --ffn_type moe \
        --num_experts 4 \
        --top_k 1 \
        --seed "$seed" \
        --epochs 40000 \
        --checkpoint_dir "$dir" \
        --no_wandb
    echo "=== Done moe seed=$seed ==="
done

echo "All extra grokking seeds done!"
