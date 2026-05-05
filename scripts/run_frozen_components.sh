#!/bin/bash
# Frozen component training on modular addition AND add-7.
# Train with frozen attention or frozen FFN for all 4 variants x 5 seeds.
cd "$(dirname "$0")/.."
# Usage: bash scripts/run_frozen_components.sh

# Auto-repair venv if torch disappears mid-run
check_venv() {
    if ! .venv/bin/python -c "import torch" 2>/dev/null; then
        echo "!!! torch missing — rebuilding venv !!!"
        rm -rf .venv
        UV_HTTP_TIMEOUT=300 uv sync 2>&1 | tail -1
    fi
}

SEEDS="42 137 256 512 1024"
FFN_TYPES="ffn glu moe moe_glu"
EPOCHS=40000
ADD7_STEPS=10000
ADD7_NUM_DIGITS=3

######################################################################
# Modular addition: frozen attention
######################################################################
for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/modadd_${ftype}_frozenattn_s${seed}"

        if [ -f "$dir/modadd_best.pt" ]; then
            echo "=== SKIP modadd $ftype frozen-attn seed=$seed (already exists) ==="
            continue
        fi

        check_venv
        echo "=== Training modadd $ftype frozen-attention seed=$seed ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_frozen.py \
            --ffn_type "$ftype" \
            --freeze attention \
            --seed "$seed" \
            --epochs $EPOCHS \
            --checkpoint_dir "$dir" \
            --log_interval 500 \
            $extra_args

        echo "=== Done modadd $ftype frozen-attention seed=$seed ==="
        echo
    done
done

######################################################################
# Modular addition: frozen FFN
######################################################################
for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/modadd_${ftype}_frozenffn_s${seed}"

        if [ -f "$dir/modadd_best.pt" ]; then
            echo "=== SKIP modadd $ftype frozen-ffn seed=$seed (already exists) ==="
            continue
        fi

        check_venv
        echo "=== Training modadd $ftype frozen-FFN seed=$seed ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_frozen.py \
            --ffn_type "$ftype" \
            --freeze ffn \
            --seed "$seed" \
            --epochs $EPOCHS \
            --checkpoint_dir "$dir" \
            --log_interval 500 \
            $extra_args

        echo "=== Done modadd $ftype frozen-ffn seed=$seed ==="
        echo
    done
done

######################################################################
# Add-7 (no norm): frozen attention
######################################################################
for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/add7_${ftype}_nonorm_frozenattn_s${seed}"

        if [ -f "$dir/best_model.pt" ]; then
            echo "=== SKIP add7 $ftype frozen-attn seed=$seed (already exists) ==="
            continue
        fi

        check_venv
        echo "=== Training add7 $ftype frozen-attention seed=$seed (no norm) ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_frozen_add7.py \
            --ffn_type "$ftype" \
            --freeze attention \
            --seed "$seed" \
            --num_digits $ADD7_NUM_DIGITS \
            --steps $ADD7_STEPS \
            --eval_interval 200 \
            --patience 50 \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            $extra_args

        echo "=== Done add7 $ftype frozen-attention seed=$seed ==="
        echo
    done
done

######################################################################
# Add-7 (no norm): frozen FFN
######################################################################
for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/add7_${ftype}_nonorm_frozenffn_s${seed}"

        if [ -f "$dir/best_model.pt" ]; then
            echo "=== SKIP add7 $ftype frozen-ffn seed=$seed (already exists) ==="
            continue
        fi

        check_venv
        echo "=== Training add7 $ftype frozen-FFN seed=$seed (no norm) ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_frozen_add7.py \
            --ffn_type "$ftype" \
            --freeze ffn \
            --seed "$seed" \
            --num_digits $ADD7_NUM_DIGITS \
            --steps $ADD7_STEPS \
            --eval_interval 200 \
            --patience 50 \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            $extra_args

        echo "=== Done add7 $ftype frozen-ffn seed=$seed ==="
        echo
    done
done

echo "All frozen-component runs complete!"
