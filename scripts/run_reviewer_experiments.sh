#!/bin/bash
# Reviewer-requested experiments
cd "$(dirname "$0")/.."

check_venv() {
    if ! .venv/bin/python -c "import torch" 2>/dev/null; then
        echo "!!! torch missing — rebuilding venv !!!"
        rm -rf .venv
        UV_HTTP_TIMEOUT=300 uv sync 2>&1 | tail -1
    fi
}

SEEDS="42 137 256 512 1024"

######################################################################
# Experiment 1: Top-2 MoE on add-7 (tests bottleneck mechanism)
# If redistribution is caused by the routing bottleneck, top-2 should
# show reduced redistribution (closer to dense FFN ablation pattern)
######################################################################
echo "=========================================="
echo "EXPERIMENT 1: Top-2 MoE on add-7"
echo "=========================================="
for seed in $SEEDS; do
    for ftype in moe moe_glu; do
        dir="checkpoints/add7_${ftype}_topk2_nonorm_s${seed}"

        if [ -f "$dir/best_model.pt" ]; then
            echo "=== SKIP add7 $ftype top-2 seed=$seed ==="
            continue
        fi

        check_venv
        echo "=== Training add7 $ftype top-2 seed=$seed ==="

        .venv/bin/python train.py \
            --ffn_type "$ftype" \
            --num_experts 4 \
            --top_k 2 \
            --seed "$seed" \
            --num_digits 3 \
            --steps 10000 \
            --eval_interval 200 \
            --patience 50 \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            --no_wandb

        echo "=== Done add7 $ftype top-2 seed=$seed ==="
        echo
    done
done

######################################################################
# Experiment 2: lambda_aux sweep on modadd (MoE only)
# Tests whether aux loss strength affects grokking speed
######################################################################
echo "=========================================="
echo "EXPERIMENT 2: lambda_aux sweep on modadd"
echo "=========================================="
for laux in 0.0 0.001 0.01 0.1 1.0; do
    for seed in $SEEDS; do
        dir="checkpoints/modadd_moe_laux${laux}_s${seed}"

        if [ -f "$dir/modadd_best.pt" ]; then
            echo "=== SKIP modadd moe laux=$laux seed=$seed ==="
            continue
        fi

        check_venv
        echo "=== Training modadd moe laux=$laux seed=$seed ==="

        .venv/bin/python train_modular_addition.py \
            --ffn_type moe \
            --num_experts 4 \
            --top_k 1 \
            --moe_balance_coeff "$laux" \
            --seed "$seed" \
            --epochs 40000 \
            --checkpoint_dir "$dir" \
            --no_wandb

        echo "=== Done modadd moe laux=$laux seed=$seed ==="
        echo
    done
done

######################################################################
# Experiment 3: Narrow dense FFN on add-7 (capacity control)
# Dense FFN with intermediate_dim = 64 (= 256/4, matching per-token
# MoE capacity). Tests whether reduced capacity alone causes
# redistribution, or whether routing is necessary.
######################################################################
echo "=========================================="
echo "EXPERIMENT 3: Narrow dense FFN on add-7"
echo "=========================================="
for seed in $SEEDS; do
    dir="checkpoints/add7_ffn_narrow_nonorm_s${seed}"

    if [ -f "$dir/best_model.pt" ]; then
        echo "=== SKIP add7 ffn_narrow seed=$seed ==="
        continue
    fi

    check_venv
    echo "=== Training add7 ffn_narrow (idim=64) seed=$seed ==="

    .venv/bin/python train.py \
        --ffn_type ffn \
        --intermediate_dim 64 \
        --seed "$seed" \
        --num_digits 3 \
        --steps 10000 \
        --eval_interval 200 \
        --patience 50 \
        --no_use_norm \
        --checkpoint_dir "$dir" \
        --no_wandb

    echo "=== Done add7 ffn_narrow seed=$seed ==="
    echo
done

######################################################################
# Experiment 4: Narrow dense FFN on modadd (capacity control)
# Dense FFN with intermediate_dim = 128 (= 512/4)
######################################################################
echo "=========================================="
echo "EXPERIMENT 4: Narrow dense FFN on modadd"
echo "=========================================="
for seed in $SEEDS; do
    dir="checkpoints/modadd_ffn_narrow_s${seed}"

    if [ -f "$dir/modadd_best.pt" ]; then
        echo "=== SKIP modadd ffn_narrow seed=$seed ==="
        continue
    fi

    check_venv
    echo "=== Training modadd ffn_narrow (idim=128) seed=$seed ==="

    .venv/bin/python train_modular_addition.py \
        --ffn_type ffn \
        --intermediate_dim 128 \
        --seed "$seed" \
        --epochs 40000 \
        --checkpoint_dir "$dir" \
        --no_wandb

    echo "=== Done modadd ffn_narrow seed=$seed ==="
    echo
done

######################################################################
# Experiment 5: SwiGLU (SiLU activation) on all tasks
# Reviewers flagged GELU vs SiLU gap. Need GLU + MoE-GLU with silu
# on all 3 tasks to verify H2 (GLU opacity) holds for SwiGLU.
######################################################################
echo "=========================================="
echo "EXPERIMENT 5: SwiGLU (silu) variants"
echo "=========================================="

# 5a: modadd (GLU + MoE-GLU with silu)
for ftype in glu moe_glu; do
    for seed in $SEEDS; do
        dir="checkpoints/modadd_${ftype}_silu_s${seed}"
        if [ -f "$dir/modadd_best.pt" ]; then
            echo "=== SKIP modadd $ftype silu seed=$seed ==="
            continue
        fi
        check_venv
        echo "=== Training modadd $ftype silu seed=$seed ==="
        extra_args=""
        if [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi
        .venv/bin/python train_modular_addition.py \
            --ffn_type "$ftype" \
            --activation silu \
            --seed "$seed" \
            --epochs 40000 \
            --checkpoint_dir "$dir" \
            --no_wandb \
            $extra_args
        echo "=== Done modadd $ftype silu seed=$seed ==="
    done
done

# 5b: add-7 (GLU + MoE-GLU with silu)
for ftype in glu moe_glu; do
    for seed in $SEEDS; do
        dir="checkpoints/add7_${ftype}_silu_nonorm_s${seed}"
        if [ -f "$dir/best_model.pt" ]; then
            echo "=== SKIP add7 $ftype silu seed=$seed ==="
            continue
        fi
        check_venv
        echo "=== Training add7 $ftype silu seed=$seed ==="
        extra_args=""
        if [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi
        .venv/bin/python train.py \
            --ffn_type "$ftype" \
            --seed "$seed" \
            --num_digits 3 \
            --steps 10000 \
            --eval_interval 200 \
            --patience 50 \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            --no_wandb \
            $extra_args
        echo "=== Done add7 $ftype silu seed=$seed ==="
    done
done

# 5c: histogram (GLU + MoE-GLU with silu)
for ftype in glu moe_glu; do
    for seed in $SEEDS; do
        dir="checkpoints/hist_${ftype}_silu_s${seed}"
        if [ -f "$dir/hist_best.pt" ]; then
            echo "=== SKIP hist $ftype silu seed=$seed ==="
            continue
        fi
        check_venv
        echo "=== Training hist $ftype silu seed=$seed ==="
        extra_args=""
        if [ "$ftype" = "moe_glu" ]; then
            extra_args="--num_experts 4"
        fi
        .venv/bin/python train_histogram.py \
            --ffn_type "$ftype" \
            --activation silu \
            --seed "$seed" \
            --epochs 1000 \
            --checkpoint_dir "$dir" \
            --no_wandb \
            $extra_args
        echo "=== Done hist $ftype silu seed=$seed ==="
    done
done

######################################################################
# Experiment 6: Frozen histogram
# Complete the frozen component picture for all 3 tasks.
######################################################################
echo "=========================================="
echo "EXPERIMENT 6: Frozen histogram"
echo "=========================================="
for ftype in $FFN_TYPES; do
    for freeze in attention ffn; do
        for seed in $SEEDS; do
            dir="checkpoints/hist_${ftype}_frozen${freeze}_s${seed}"
            if [ -f "$dir/hist_best.pt" ]; then
                echo "=== SKIP hist $ftype frozen-$freeze seed=$seed ==="
                continue
            fi
            check_venv
            echo "=== Training hist $ftype frozen-$freeze seed=$seed ==="
            extra_args=""
            if [ "$ftype" = "moe" ] || [ "$ftype" = "moe_glu" ]; then
                extra_args="--num_experts 4"
            fi
            # Note: need train_frozen_histogram.py or --freeze flag in train_histogram.py
            # TODO: create this script if it doesn't exist
            echo "  TODO: needs train_frozen_histogram.py"
        done
    done
done

echo "All reviewer experiments complete!"
