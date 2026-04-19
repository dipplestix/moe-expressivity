#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
for ftype in moe moe_glu; do
    for seed in $SEEDS; do
        dir="checkpoints/add7_${ftype}_randroute_nonorm_s${seed}"
        if [ -f "$dir/best_model.pt" ]; then
            echo "=== SKIP add7 $ftype randroute seed=$seed ==="
            continue
        fi
        echo "=== Training add7 $ftype randroute seed=$seed ==="
        .venv/bin/python train.py \
            --ffn_type "$ftype" \
            --num_experts 4 \
            --top_k 1 \
            --random_routing \
            --seed "$seed" \
            --num_digits 3 \
            --steps 10000 \
            --eval_interval 200 \
            --patience 50 \
            --no_use_norm \
            --checkpoint_dir "$dir" \
            --no_wandb
        echo "=== Done add7 $ftype randroute seed=$seed ==="
    done
done
echo "All add-7 random routing done!"
