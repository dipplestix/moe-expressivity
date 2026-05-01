#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
# Add-7: model_dim=64, dense FFN intermediate_dim=256, dense FFN params = 2*64*256 = 32768.
# Total-parameter-matched GLU: h_glu = (2/3)*h_dense = (2/3)*256 = 170.67 -> 170.
# Check: 3*64*170 = 32640 params, matching dense 32768 within 0.4%.
echo "=== Param-matched GLU (intermediate_dim=170) on add-7 ==="
echo "Start: $(date)"
for seed in $SEEDS; do
    dir="checkpoints/add7_glu_d170_nonorm_s${seed}"
    if [ -f "$dir/best_model.pt" ]; then
        echo "=== SKIP seed=$seed (already exists) ==="
        continue
    fi
    echo "=== Training add7 glu d=170 seed=$seed @ $(date) ==="
    .venv/bin/python train.py \
        --ffn_type glu \
        --intermediate_dim 170 \
        --seed "$seed" \
        --num_digits 3 \
        --steps 10000 \
        --eval_interval 200 \
        --patience 50 \
        --no_use_norm \
        --checkpoint_dir "$dir" \
        --no_wandb
    echo "=== Done seed=$seed @ $(date) ==="
done
echo "=== ALL add-7 param-matched GLU done @ $(date) ==="
