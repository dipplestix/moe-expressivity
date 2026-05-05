"""
SiLU vs GELU component ablation comparison.
Checks whether MoE redistribution effect holds under SiLU activation.
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ".")
sys.path.insert(0, "model")

import numpy as np
import torch
from model import OneLayerTransformer
from data.modular_addition import ModularAdditionDataset

DEVICE = "cpu"
SEEDS = [42, 137, 256, 512, 1024]
NUM_DIGITS = 3
EOS_TOKEN = 11
P = 113


def num_to_reversed_digits(n, length):
    d = []
    for _ in range(length):
        d.append(n % 10)
        n //= 10
    return d


def generate_add7_examples():
    max_val = 10 ** NUM_DIGITS - 1
    ond = NUM_DIGITS + 1
    examples = []
    for n in range(max_val + 1):
        result = n + 7
        ind = num_to_reversed_digits(n, NUM_DIGITS)
        outd = num_to_reversed_digits(result, ond)
        examples.append({'seq': ind + [EOS_TOKEN] + outd + [EOS_TOKEN]})
    return examples


def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    config = dict(ckpt["config"])
    sd = ckpt["model_state_dict"]
    # Infer intermediate_dim from state dict to avoid config/weights mismatch
    ffn_type = config.get("ffn_type", "ffn")
    if ffn_type == "glu" and "ffn.gate_proj.weight" in sd:
        config["intermediate_dim"] = sd["ffn.gate_proj.weight"].shape[0]
    elif ffn_type == "moe_glu" and "ffn.experts.0.gate_proj.weight" in sd:
        num_experts = config.get("num_experts", 4)
        config["intermediate_dim"] = sd["ffn.experts.0.gate_proj.weight"].shape[0] * num_experts
    elif ffn_type == "ffn" and "ffn.w1.weight" in sd:
        config["intermediate_dim"] = sd["ffn.w1.weight"].shape[0]
    elif ffn_type == "moe" and "ffn.experts.0.w1.weight" in sd:
        num_experts = config.get("num_experts", 4)
        config["intermediate_dim"] = sd["ffn.experts.0.w1.weight"].shape[0] * num_experts
    model = OneLayerTransformer(**config).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model


def ablation_accuracy(model, x, targets, out_start):
    """Compute normal, no-attn, no-ffn accuracy for add7."""
    with torch.no_grad():
        logits = model(x)
    normal = (logits.argmax(-1)[:, out_start:] == targets[:, out_start:]).float().mean().item()

    orig_attn = model.atn.forward
    model.atn.forward = lambda *a, _o=orig_attn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_na = model(x)
    no_attn = (logits_na.argmax(-1)[:, out_start:] == targets[:, out_start:]).float().mean().item()
    model.atn.forward = orig_attn

    orig_ffn = model.ffn.forward
    model.ffn.forward = lambda *a, _o=orig_ffn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_nf = model(x)
    no_ffn = (logits_nf.argmax(-1)[:, out_start:] == targets[:, out_start:]).float().mean().item()
    model.ffn.forward = orig_ffn

    return normal, no_attn, no_ffn


def modadd_ablation_accuracy(model, inputs, targets):
    """Compute normal, no-attn, no-ffn accuracy for modular addition."""
    with torch.no_grad():
        logits = model(inputs)
    normal = (logits[:, 2, :].argmax(-1) == targets).float().mean().item()

    orig_attn = model.atn.forward
    model.atn.forward = lambda *a, _o=orig_attn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_na = model(inputs)
    no_attn = (logits_na[:, 2, :].argmax(-1) == targets).float().mean().item()
    model.atn.forward = orig_attn

    orig_ffn = model.ffn.forward
    model.ffn.forward = lambda *a, _o=orig_ffn, **k: torch.zeros_like(_o(*a, **k))
    with torch.no_grad():
        logits_nf = model(inputs)
    no_ffn = (logits_nf[:, 2, :].argmax(-1) == targets).float().mean().item()
    model.ffn.forward = orig_ffn

    return normal, no_attn, no_ffn


def get_all_modadd_inputs(p=P):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    inputs = torch.stack([a, b, eq], dim=1)
    targets = (a + b) % p
    return inputs, targets


def fmt(vals):
    """Format mean +/- std as string."""
    return f"{np.mean(vals)*100:.1f} +/- {np.std(vals)*100:.1f}"


def run_add7():
    print("=" * 80)
    print("ADD-7 COMPONENT ABLATION: GELU vs SiLU")
    print("=" * 80)

    examples = generate_add7_examples()
    seqs = torch.tensor([e['seq'] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS + 1

    configs = {
        "GLU (GELU)":       ("checkpoints/add7_glu_nonorm_s{seed}/best_model.pt", SEEDS),
        "GLU (SiLU)":       ("checkpoints/add7_glu_silu_nonorm_s{seed}/best_model.pt", SEEDS),
        "MoE-GLU (GELU)":   ("checkpoints/add7_moe_glu_nonorm_s{seed}/best_model.pt", SEEDS),
        "MoE-GLU (SiLU)":   ("checkpoints/add7_moe_glu_silu_nonorm_s{seed}/best_model.pt", SEEDS),
    }

    all_results = {}
    for label, (path_template, seeds) in configs.items():
        normals, no_attns, no_ffns = [], [], []
        for seed in seeds:
            path = path_template.format(seed=seed)
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping")
                continue
            model = load_model(path)
            normal, no_attn, no_ffn = ablation_accuracy(model, x_input, targets, out_start)
            normals.append(normal)
            no_attns.append(no_attn)
            no_ffns.append(no_ffn)
        all_results[label] = (normals, no_attns, no_ffns)
        print(f"  {label}: {len(normals)} seeds loaded")

    print()
    print(f"{'Model':<20} {'Normal':>18} {'No-Attn':>18} {'No-FFN':>18}")
    print("-" * 76)
    for label in configs:
        normals, no_attns, no_ffns = all_results[label]
        print(f"{label:<20} {fmt(normals):>18} {fmt(no_attns):>18} {fmt(no_ffns):>18}")

    # Key comparison
    print()
    print("KEY COMPARISON: Does MoE redistribute computation under SiLU?")
    for act in ["GELU", "SiLU"]:
        glu_ffn = all_results[f"GLU ({act})"][2]  # no_ffn values
        moe_ffn = all_results[f"MoE-GLU ({act})"][2]
        glu_attn = all_results[f"GLU ({act})"][1]  # no_attn values
        moe_attn = all_results[f"MoE-GLU ({act})"][1]
        print(f"  {act}: GLU no-FFN={np.mean(glu_ffn)*100:.1f}%, MoE-GLU no-FFN={np.mean(moe_ffn)*100:.1f}% "
              f"(delta={np.mean(moe_ffn)*100 - np.mean(glu_ffn)*100:+.1f}pp)")
        print(f"  {act}: GLU no-Attn={np.mean(glu_attn)*100:.1f}%, MoE-GLU no-Attn={np.mean(moe_attn)*100:.1f}% "
              f"(delta={np.mean(moe_attn)*100 - np.mean(glu_attn)*100:+.1f}pp)")


def run_modadd():
    print()
    print("=" * 80)
    print("MODULAR ADDITION COMPONENT ABLATION: GELU vs SiLU")
    print("=" * 80)

    inputs, targets = get_all_modadd_inputs()

    # For GLU SiLU, skip seed 1024 (failed training)
    glu_silu_seeds = [42, 137, 256, 512]

    configs = {
        "GLU (GELU)":       ("checkpoints/modadd_glu_s{seed}/modadd_best.pt", SEEDS),
        "GLU (SiLU)":       ("checkpoints/modadd_glu_silu_s{seed}/modadd_best.pt", glu_silu_seeds),
        "MoE-GLU (GELU)":   ("checkpoints/modadd_moe_glu_s{seed}/modadd_best.pt", SEEDS),
        "MoE-GLU (SiLU)":   ("checkpoints/modadd_moe_glu_silu_s{seed}/modadd_best.pt", SEEDS),
    }

    all_results = {}
    for label, (path_template, seeds) in configs.items():
        normals, no_attns, no_ffns = [], [], []
        for seed in seeds:
            path = path_template.format(seed=seed)
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping")
                continue
            model = load_model(path)
            normal, no_attn, no_ffn = modadd_ablation_accuracy(model, inputs, targets)
            normals.append(normal)
            no_attns.append(no_attn)
            no_ffns.append(no_ffn)
        all_results[label] = (normals, no_attns, no_ffns)
        n_seeds = len(normals)
        note = f" (skipped s1024)" if label == "GLU (SiLU)" else ""
        print(f"  {label}: {n_seeds} seeds loaded{note}")

    print()
    print(f"{'Model':<20} {'Normal':>18} {'No-Attn':>18} {'No-FFN':>18}")
    print("-" * 76)
    for label in configs:
        normals, no_attns, no_ffns = all_results[label]
        print(f"{label:<20} {fmt(normals):>18} {fmt(no_attns):>18} {fmt(no_ffns):>18}")

    print()
    print("KEY COMPARISON: Does MoE redistribute computation under SiLU?")
    for act in ["GELU", "SiLU"]:
        glu_ffn = all_results[f"GLU ({act})"][2]
        moe_ffn = all_results[f"MoE-GLU ({act})"][2]
        glu_attn = all_results[f"GLU ({act})"][1]
        moe_attn = all_results[f"MoE-GLU ({act})"][1]
        print(f"  {act}: GLU no-FFN={np.mean(glu_ffn)*100:.1f}%, MoE-GLU no-FFN={np.mean(moe_ffn)*100:.1f}% "
              f"(delta={np.mean(moe_ffn)*100 - np.mean(glu_ffn)*100:+.1f}pp)")
        print(f"  {act}: GLU no-Attn={np.mean(glu_attn)*100:.1f}%, MoE-GLU no-Attn={np.mean(moe_attn)*100:.1f}% "
              f"(delta={np.mean(moe_attn)*100 - np.mean(glu_attn)*100:+.1f}pp)")


if __name__ == "__main__":
    run_add7()
    run_modadd()
