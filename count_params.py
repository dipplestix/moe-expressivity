#!/usr/bin/env -S uv run
"""Script to count and display parameters for each component of OneLayerTransformer."""

import torch
import sys
sys.path.insert(0, 'model')

from model import OneLayerTransformer


def count_parameters(module):
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def format_params(n):
    """Format parameter count with commas and human-readable suffix."""
    if n >= 1_000_000:
        return f"{n:,} ({n/1_000_000:.2f}M)"
    elif n >= 1_000:
        return f"{n:,} ({n/1_000:.2f}K)"
    return f"{n:,}"


def analyze_model(model_dim=256, num_heads=8, ffn_type="ffn", vocab_size=10, max_seq_len=128):
    """Analyze parameter counts for OneLayerTransformer."""

    print("=" * 70)
    print(f"OneLayerTransformer Parameter Analysis")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  model_dim:    {model_dim}")
    print(f"  num_heads:    {num_heads}")
    print(f"  ffn_type:     {ffn_type}")
    print(f"  vocab_size:   {vocab_size}")
    print(f"  max_seq_len:  {max_seq_len}")
    print()

    model = OneLayerTransformer(
        model_dim=model_dim,
        num_heads=num_heads,
        ffn_type=ffn_type,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )

    # Component breakdown
    components = {
        "vocab (token embedding)": model.vocab,
        "pos_embed (positional embedding)": model.pos_embed,
        "atn_norm (pre-attention RMSNorm)": model.atn_norm,
        "atn (multi-head attention)": model.atn,
        "ffn_norm (pre-FFN RMSNorm)": model.ffn_norm,
        f"ffn ({ffn_type.upper()})": model.ffn,
        "out_norm (output RMSNorm)": model.out_norm,
    }

    print("-" * 70)
    print(f"{'Component':<40} {'Parameters':>25}")
    print("-" * 70)

    total = 0
    for name, module in components.items():
        params = count_parameters(module)
        total += params
        print(f"{name:<40} {format_params(params):>25}")

    print("-" * 70)
    print(f"{'TOTAL':<40} {format_params(total):>25}")
    print("-" * 70)

    # Detailed breakdown for attention
    print("\n" + "=" * 70)
    print("Attention (MHA) Breakdown:")
    print("-" * 70)
    atn_components = {
        "q_proj": model.atn.q_proj,
        "k_proj": model.atn.k_proj,
        "v_proj": model.atn.v_proj,
        "o_proj": model.atn.o_proj,
    }
    for name, module in atn_components.items():
        params = count_parameters(module)
        print(f"  {name:<36} {format_params(params):>25}")

    # Detailed breakdown for FFN
    print("\n" + "=" * 70)
    print(f"FFN ({ffn_type.upper()}) Breakdown:")
    print("-" * 70)
    if ffn_type == "ffn":
        ffn_components = {
            "up_proj": model.ffn.up_proj,
            "down_proj": model.ffn.down_proj,
        }
    else:  # glu
        ffn_components = {
            "gate_proj": model.ffn.gate_proj,
            "up_proj": model.ffn.up_proj,
            "down_proj": model.ffn.down_proj,
        }
    for name, module in ffn_components.items():
        params = count_parameters(module)
        print(f"  {name:<36} {format_params(params):>25}")

    # Parameter distribution
    print("\n" + "=" * 70)
    print("Parameter Distribution:")
    print("-" * 70)

    categories = {
        "Embeddings": count_parameters(model.vocab) + count_parameters(model.pos_embed),
        "Attention": count_parameters(model.atn),
        "FFN": count_parameters(model.ffn),
        "Normalization": (count_parameters(model.atn_norm) +
                         count_parameters(model.ffn_norm) +
                         count_parameters(model.out_norm)),
    }

    for name, params in categories.items():
        pct = 100 * params / total
        bar = "#" * int(pct / 2)
        print(f"  {name:<15} {format_params(params):>20} ({pct:5.1f}%) {bar}")

    print("\n")
    return model


if __name__ == "__main__":
    # Default analysis
    print("\n" + "=" * 70)
    print("ANALYSIS WITH FFN")
    analyze_model(ffn_type="ffn")

    print("\n" + "=" * 70)
    print("ANALYSIS WITH GLU")
    analyze_model(ffn_type="glu")
