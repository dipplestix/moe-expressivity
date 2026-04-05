#!/bin/bash
# Frozen component training on modular addition.
# Train with frozen attention or frozen FFN for all 4 variants x 5 seeds.
cd "$(dirname "$0")/.."

SEEDS="42 137 256 512 1024"
FFN_TYPES="ffn glu moe moe_glu"
EPOCHS=40000

echo "=== Frozen Component Training ==="
echo

# Frozen attention (only FFN learns)
for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/modadd_${ftype}_frozenattn_s${seed}"
        if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP $ftype frozen-attn s$seed"; continue; fi
        echo "Training $ftype frozen-attention seed=$seed"

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_frozen.py \
            --ffn_type "$ftype" --freeze attention --seed "$seed" \
            --epochs $EPOCHS --checkpoint_dir "$dir" \
            --log_interval 500 $extra_args
    done
done

# Frozen FFN (only attention learns)
for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/modadd_${ftype}_frozenffn_s${seed}"
        if [ -f "$dir/modadd_best.pt" ]; then echo "SKIP $ftype frozen-ffn s$seed"; continue; fi
        echo "Training $ftype frozen-FFN seed=$seed"

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_frozen.py \
            --ffn_type "$ftype" --freeze ffn --seed "$seed" \
            --epochs $EPOCHS --checkpoint_dir "$dir" \
            --log_interval 500 $extra_args
    done
done

echo "Frozen component training complete!"
