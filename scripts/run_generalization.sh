#!/bin/bash
# Held-out generalization experiments for add-7 (3-digit).
# Three experiments x 4 variants x 5 seeds = 60 runs.
cd "$(dirname "$0")/.."

# Auto-repair venv if torch disappears
check_venv() {
    if ! .venv/bin/python -c "import torch" 2>/dev/null; then
        echo "!!! torch missing — rebuilding venv !!!"
        rm -rf .venv
        UV_HTTP_TIMEOUT=300 uv sync 2>&1 | tail -1
    fi
}

SEEDS="42 137 256 512 1024"
FFN_TYPES="ffn glu moe moe_glu"
STEPS=10000
NUM_DIGITS=3

######################################################################
# Experiment 1: Exclude carry length >= 2
# Train only on L=0,1; test generalization to L=2,3
# 3 digits: ~70 held-out numbers (L=2: 63, L=3: 7)
######################################################################
echo "=========================================="
echo "EXPERIMENT 1: Exclude carry >= 2 (3-digit)"
echo "=========================================="
for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/gen_excludecarry2_${ftype}_s${seed}"

        if [ -f "$dir/generalization_results.json" ]; then
            echo "=== SKIP exclude_carry_ge2 $ftype seed=$seed (results exist) ==="
            continue
        fi

        check_venv
        echo "=== Training exclude_carry_ge2 $ftype seed=$seed ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_generalization.py \
            --ffn_type "$ftype" \
            --exclude_carry_ge 2 \
            --seed "$seed" \
            --num_digits $NUM_DIGITS \
            --steps $STEPS \
            --eval_interval 200 \
            --patience 50 \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            --no_wandb \
            $extra_args

        echo "=== Done exclude_carry_ge2 $ftype seed=$seed ==="
        echo
    done
done

######################################################################
# Experiment 2: Exclude ones digit >= 3 (no-carry training)
# Train only on inputs that don't trigger carry; test on carry inputs
# 3 digits: 300 train, 700 held-out
######################################################################
echo "=========================================="
echo "EXPERIMENT 2: Exclude ones >= 3 (3-digit)"
echo "=========================================="
for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/gen_excludeones3_${ftype}_s${seed}"

        if [ -f "$dir/generalization_results.json" ]; then
            echo "=== SKIP exclude_ones_ge3 $ftype seed=$seed (results exist) ==="
            continue
        fi

        check_venv
        echo "=== Training exclude_ones_ge3 $ftype seed=$seed ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_generalization.py \
            --ffn_type "$ftype" \
            --exclude_ending_9 \
            --seed "$seed" \
            --num_digits $NUM_DIGITS \
            --steps $STEPS \
            --eval_interval 200 \
            --patience 50 \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            --no_wandb \
            $extra_args

        echo "=== Done exclude_ones_ge3 $ftype seed=$seed ==="
        echo
    done
done

######################################################################
# Experiment 3: Exclude tens digit = 9
# Model sees +1 at tens (single carry) but never carry propagation
# through tens=9 to hundreds. Tests positional generalization of +1.
# 3 digits: 900 train, 100 held-out
######################################################################
echo "=========================================="
echo "EXPERIMENT 3: Exclude tens=9 (3-digit)"
echo "=========================================="
for seed in $SEEDS; do
    for ftype in $FFN_TYPES; do
        dir="checkpoints/gen_excludetens9_${ftype}_s${seed}"

        if [ -f "$dir/generalization_results.json" ]; then
            echo "=== SKIP exclude_tens9 $ftype seed=$seed (results exist) ==="
            continue
        fi

        check_venv
        echo "=== Training exclude_tens9 $ftype seed=$seed ==="

        extra_args=""
        if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi

        .venv/bin/python train_generalization.py \
            --ffn_type "$ftype" \
            --exclude_tens_9 \
            --seed "$seed" \
            --num_digits $NUM_DIGITS \
            --steps $STEPS \
            --eval_interval 200 \
            --patience 50 \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            --no_wandb \
            $extra_args

        echo "=== Done exclude_tens9 $ftype seed=$seed ==="
        echo
    done
done

echo "All generalization experiments complete!"
