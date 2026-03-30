"""
H1 component ablation for modular addition across all 4 variants x 5 seeds.
Zero attention output vs zero FFN output, measure test accuracy.
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "model")
from model import OneLayerTransformer
from data.modular_addition import ModularAdditionDataset

P = 113
DEVICE = "cpu"


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    config = ckpt["config"]
    model = OneLayerTransformer(**config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def get_all_inputs(p=P):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    inputs = torch.stack([a, b, eq], dim=1)
    targets = (a + b) % p
    return inputs, targets


def evaluate(model, inputs, targets):
    with torch.no_grad():
        logits = model(inputs)
    preds = logits[:, 2, :].argmax(dim=-1)
    return (preds == targets).float().mean().item()


def ablation_study(model, inputs, targets):
    """Zero attention vs zero FFN, measure accuracy."""
    normal_acc = evaluate(model, inputs, targets)

    # Zero attention
    orig_attn = model.atn.forward
    model.atn.forward = lambda *args, **kwargs: torch.zeros_like(orig_attn(*args, **kwargs))
    no_attn_acc = evaluate(model, inputs, targets)
    model.atn.forward = orig_attn

    # Zero FFN
    orig_ffn = model.ffn.forward
    model.ffn.forward = lambda *args, **kwargs: torch.zeros_like(orig_ffn(*args, **kwargs))
    no_ffn_acc = evaluate(model, inputs, targets)
    model.ffn.forward = orig_ffn

    return normal_acc, no_attn_acc, no_ffn_acc


if __name__ == "__main__":
    inputs, targets = get_all_inputs()
    print(f"Evaluating on all {P}^2 = {len(inputs)} pairs\n")

    ftypes = ['ffn', 'glu', 'moe', 'moe_glu']
    seeds = [42, 137, 256, 512, 1024]

    all_results = {}

    for ftype in ftypes:
        print(f"{'='*50}")
        print(f"  {ftype.upper()}")
        print(f"{'='*50}")

        normals, no_attns, no_ffns = [], [], []

        for seed in seeds:
            path = f"checkpoints/modadd_{ftype}_s{seed}/modadd_best.pt"
            model, ckpt = load_model(path)
            normal, no_attn, no_ffn = ablation_study(model, inputs, targets)
            normals.append(normal)
            no_attns.append(no_attn)
            no_ffns.append(no_ffn)
            print(f"  s{seed}: normal={normal:.4f} no_attn={no_attn:.4f} no_ffn={no_ffn:.4f}")

        n = np.array(normals)
        a = np.array(no_attns)
        f = np.array(no_ffns)
        print(f"\n  Aggregated:")
        print(f"    Normal:  {n.mean():.4f} +/- {n.std():.4f}")
        print(f"    No attn: {a.mean():.4f} +/- {a.std():.4f}")
        print(f"    No FFN:  {f.mean():.4f} +/- {f.std():.4f}")
        print()

        all_results[ftype] = {
            'normal': (n.mean(), n.std()),
            'no_attn': (a.mean(), a.std()),
            'no_ffn': (f.mean(), f.std()),
        }

    # Summary table
    print(f"\n{'='*50}")
    print(f"  SUMMARY: Component Ablation on (a+b) mod {P}")
    print(f"{'='*50}")
    print(f"{'Variant':<12} {'Normal':>10} {'No Attn':>14} {'No FFN':>14}")
    print(f"{'-'*50}")
    for ftype in ftypes:
        r = all_results[ftype]
        print(f"{ftype:<12} {r['normal'][0]:>8.4f}   "
              f"{r['no_attn'][0]:>7.4f}+/-{r['no_attn'][1]:.4f} "
              f"{r['no_ffn'][0]:>7.4f}+/-{r['no_ffn'][1]:.4f}")

    # Comparison with add-7 (hardcoded from previous analysis)
    print(f"\n{'='*50}")
    print(f"  COMPARISON: Modular Addition vs Add-7")
    print(f"{'='*50}")
    print(f"{'Variant':<12} {'ModAdd NoAttn':>14} {'ModAdd NoFFN':>14} {'Add7 NoAttn':>14} {'Add7 NoFFN':>14}")
    print(f"{'-'*70}")
    add7 = {
        'ffn':     (0.548, 0.231),
        'glu':     (0.548, 0.278),
        'moe':     (0.363, 0.951),
        'moe_glu': (0.502, 0.576),
    }
    for ftype in ftypes:
        r = all_results[ftype]
        a7 = add7[ftype]
        print(f"{ftype:<12} {r['no_attn'][0]:>12.4f}   {r['no_ffn'][0]:>12.4f}   "
              f"{a7[0]:>12.4f}   {a7[1]:>12.4f}")
