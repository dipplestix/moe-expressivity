"""
FormerLens: Hooked transformer components for interpretability analysis.

This module provides TransformerLens-compatible hooked versions of the
OneLayerTransformer model, enabling detailed inspection of intermediate
activations during forward passes.

Usage:
    from formerlens import HookedOneLayerTransformer
    
    # Create a new hooked model
    model = HookedOneLayerTransformer(model_dim=64, num_heads=4)
    
    # Or load from an existing model
    from model.model import OneLayerTransformer
    original = OneLayerTransformer(model_dim=64, num_heads=4)
    hooked = HookedOneLayerTransformer.from_pretrained(original)
    
    # Or load from a checkpoint
    hooked = HookedOneLayerTransformer.from_checkpoint("checkpoints/best_model.pt")
    
    # Run with cache to capture all activations
    logits, cache = hooked.run_with_cache(tokens)
    
    # Access specific activations
    attn_pattern = cache["atn.hook_pattern"]
    ffn_activations = cache["ffn.hook_post_act"]
"""

from .hooked_former import HookedOneLayerTransformer
from .hooked_components import HookedFFN, HookedGLU, HookedMHA

__all__ = [
    "HookedOneLayerTransformer",
    "HookedFFN",
    "HookedGLU", 
    "HookedMHA",
]

