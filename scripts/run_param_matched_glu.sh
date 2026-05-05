#!/bin/bash
cd "$(dirname "$0")/.."
SEEDS="42 137 256 512 1024"
# For modadd: model_dim=128, dense FFN intermediate=512 (131k params)
# GLU param-matched needs intermediate=340 (~130k FFN params, 1% off)
echo "=== Param-matched GLU (intermediate_dim=340) on modadd ==="
for seed in $SEEDS; do
    dir="checkpoints/modadd_glu_d340_s${seed}"
    if [ -f "$dir/modadd_best.pt" ]; then
        echo "=== SKIP seed=$seed ==="
        continue
    fi
    echo "=== Training modadd glu d=340 seed=$seed ==="
    .venv/bin/python train_modular_addition.py \
        --ffn_type glu \
        --intermediate_dim 340 \
        --seed "$seed" \
        --epochs 40000 \
        --checkpoint_dir "$dir" \
        --no_wandb
    echo "=== Done seed=$seed ==="
done
echo "All param-matched GLU done!"
