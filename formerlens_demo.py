"""
Demo script showing how to use HookedOneLayerTransformer for interpretability analysis.
"""

import torch
import sys
import os

# Add model directory to path for component imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from formerlens import HookedOneLayerTransformer

# Import after path setup
from model import OneLayerTransformer


def demo_basic_usage():
    """Create a hooked model and run with cache."""
    print("=" * 60)
    print("Basic Usage: Create and run hooked model")
    print("=" * 60)
    
    # Create hooked model directly
    model = HookedOneLayerTransformer(
        model_dim=64,
        num_heads=4,
        ffn_type="ffn",
        vocab_size=100,
    )
    model.eval()
    
    # Print available hooks
    model.print_hooks()
    
    # Create dummy input
    tokens = torch.randint(0, 100, (2, 16))  # batch=2, seq_len=16
    
    # Run with cache to capture all intermediate activations
    logits, cache = model.run_with_cache(tokens)
    
    print(f"\nInput shape: {tokens.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"\nCached activations:")
    for name, tensor in cache.items():
        print(f"  {name}: {tensor.shape}")


def demo_from_pretrained():
    """Load hooked model from an existing trained model."""
    print("\n" + "=" * 60)
    print("From Pretrained: Convert existing model to hooked version")
    print("=" * 60)
    
    # Create original model
    original = OneLayerTransformer(
        model_dim=64,
        num_heads=4,
        ffn_type="glu",  # Using GLU to show different hooks
        vocab_size=100,
    )
    original.eval()
    
    # Convert to hooked version
    hooked = HookedOneLayerTransformer.from_pretrained(original)
    hooked.eval()
    
    # Verify outputs match
    tokens = torch.randint(0, 100, (1, 8))
    
    with torch.no_grad():
        original_logits = original(tokens)
        hooked_logits = hooked(tokens)
    
    diff = (original_logits - hooked_logits).abs().max().item()
    print(f"Max difference between original and hooked outputs: {diff:.2e}")
    print("✓ Outputs match!" if diff < 1e-5 else "✗ Outputs differ!")


def demo_attention_analysis():
    """Analyze attention patterns."""
    print("\n" + "=" * 60)
    print("Attention Analysis: Inspecting attention patterns")
    print("=" * 60)
    
    model = HookedOneLayerTransformer(
        model_dim=32,
        num_heads=2,
        vocab_size=50,
    )
    model.eval()
    
    tokens = torch.randint(0, 50, (1, 10))
    
    logits, cache = model.run_with_cache(tokens)
    
    # Get attention pattern
    attn_pattern = cache["atn.hook_pattern"]  # (batch, heads, seq, seq)
    print(f"Attention pattern shape: {attn_pattern.shape}")
    print(f"  - Batch: {attn_pattern.shape[0]}")
    print(f"  - Heads: {attn_pattern.shape[1]}")
    print(f"  - Query positions: {attn_pattern.shape[2]}")
    print(f"  - Key positions: {attn_pattern.shape[3]}")
    
    # Check attention sums to 1 (should be softmax output)
    attn_sum = attn_pattern.sum(dim=-1)
    print(f"\nAttention weights sum per position (should be ~1.0):")
    print(f"  Mean: {attn_sum.mean():.4f}, Min: {attn_sum.min():.4f}, Max: {attn_sum.max():.4f}")
    
    # Show which positions each head attends to most at last position
    print(f"\nHead attention at last token (position 9) to all previous positions:")
    for head in range(attn_pattern.shape[1]):
        pattern = attn_pattern[0, head, -1, :]  # last query position
        top_k = pattern.topk(3)
        print(f"  Head {head}: top positions = {top_k.indices.tolist()}, weights = {[f'{w:.3f}' for w in top_k.values.tolist()]}")


def demo_ffn_activations():
    """Analyze FFN neuron activations."""
    print("\n" + "=" * 60)
    print("FFN Analysis: Inspecting neuron activations")
    print("=" * 60)
    
    model = HookedOneLayerTransformer(
        model_dim=32,
        num_heads=2,
        ffn_type="ffn",
        vocab_size=50,
    )
    model.eval()
    
    tokens = torch.randint(0, 50, (1, 8))
    
    logits, cache = model.run_with_cache(tokens)
    
    # Get FFN activations
    pre_act = cache["ffn.hook_pre_act"]  # Before SiLU
    post_act = cache["ffn.hook_post_act"]  # After SiLU
    
    print(f"Pre-activation shape: {pre_act.shape}")
    print(f"Post-activation shape: {post_act.shape}")
    
    # Analyze neuron firing
    neuron_activations = post_act[0].mean(dim=0)  # Average over positions
    print(f"\nNeuron activation stats (averaged over positions):")
    print(f"  Mean: {neuron_activations.mean():.4f}")
    print(f"  Std: {neuron_activations.std():.4f}")
    print(f"  Min: {neuron_activations.min():.4f}")
    print(f"  Max: {neuron_activations.max():.4f}")
    
    # Find most active neurons
    top_neurons = neuron_activations.topk(5)
    print(f"\nTop 5 most active neurons: {top_neurons.indices.tolist()}")
    print(f"  Activations: {[f'{a:.3f}' for a in top_neurons.values.tolist()]}")


def demo_custom_hooks():
    """Run with custom hook functions."""
    print("\n" + "=" * 60)
    print("Custom Hooks: Modify activations during forward pass")
    print("=" * 60)
    
    model = HookedOneLayerTransformer(
        model_dim=32,
        num_heads=2,
        vocab_size=50,
    )
    model.eval()
    
    tokens = torch.randint(0, 50, (1, 8))
    
    # Hook that prints activation info
    def print_hook(activation, hook):
        print(f"  {hook.name}: shape={activation.shape}, mean={activation.mean():.4f}")
        return activation
    
    print("Running with print hooks on key activations:\n")
    hooks = [
        ("hook_embed", print_hook),
        ("atn.hook_pattern", print_hook),
        ("ffn.hook_post_act", print_hook),
        ("hook_logits", print_hook),
    ]
    
    with model.hooks(fwd_hooks=hooks):
        logits = model(tokens)
    
    # Hook that modifies activations (ablation example)
    def zero_head_hook(activation, hook):
        # Zero out head 0's attention
        activation[:, 0, :, :] = 0
        # Renormalize
        activation = activation / activation.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return activation
    
    print("\n\nAblation: Zero out attention head 0")
    normal_logits = model(tokens)
    
    with model.hooks(fwd_hooks=[("atn.hook_pattern", zero_head_hook)]):
        ablated_logits = model(tokens)
    
    diff = (normal_logits - ablated_logits).abs().mean().item()
    print(f"Mean absolute difference after ablating head 0: {diff:.4f}")


def demo_from_checkpoint():
    """Load from checkpoint (if exists)."""
    print("\n" + "=" * 60)
    print("From Checkpoint: Load trained model for analysis")
    print("=" * 60)
    
    import os
    checkpoint_path = "checkpoints/best_model.pt"
    
    if os.path.exists(checkpoint_path):
        try:
            model = HookedOneLayerTransformer.from_checkpoint(checkpoint_path)
            model.eval()
            print(f"✓ Loaded model from {checkpoint_path}")
            print(f"  Model dim: {model.model_dim}")
            print(f"  Num heads: {model.num_heads}")
            print(f"  FFN type: {model.ffn_type}")
            print(f"  Vocab size: {model.vocab_size}")
            
            # Run a quick test with the add-7 task format
            # Token format: reversed input digits + EOS(11) + reversed output digits + EOS(11)
            # Example: 42 + 7 = 49 -> [2, 4, 11, 9, 4, 0, 11]
            EOS_TOKEN = 11
            test_input = torch.tensor([[2, 4, EOS_TOKEN]])  # "42" reversed + EOS
            
            logits, cache = model.run_with_cache(test_input)
            print(f"\n  Test input (42 reversed + EOS): {test_input.tolist()}")
            print(f"  Logits shape: {logits.shape}")
            
            # Show what the model predicts
            next_token = logits[0, -1].argmax().item()
            print(f"  Next token prediction: {next_token} (should be 9 for 49)")
            
            # Show attention pattern for this input
            attn = cache["atn.hook_pattern"][0]  # (heads, seq, seq)
            print(f"\n  Attention patterns (each head, last position attending to all):")
            for h in range(attn.shape[0]):
                pattern = attn[h, -1, :].tolist()
                print(f"    Head {h}: {[f'{p:.2f}' for p in pattern]}")
                
        except Exception as e:
            import traceback
            print(f"Could not load checkpoint: {e}")
            traceback.print_exc()
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Train a model first using train.py")


if __name__ == "__main__":
    demo_basic_usage()
    demo_from_pretrained()
    demo_attention_analysis()
    demo_ffn_activations()
    demo_custom_hooks()
    demo_from_checkpoint()
    
    print("\n" + "=" * 60)
    print("Demo complete! Use these patterns to analyze your trained models.")
    print("=" * 60)

