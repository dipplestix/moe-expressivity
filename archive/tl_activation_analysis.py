import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # FFN Activation Analysis with TransformerLens

    This notebook uses TransformerLens to analyze FFN activations in the Add-7 model.
    TransformerLens provides powerful hooks and caching for interpretability research.

    We'll:
    - Create a HookedTransformer matching our trained model's architecture
    - Load the trained weights
    - Use TransformerLens's caching to analyze activations
    - Compare overflow vs non-overflow digit patterns
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.utils import get_act_name
    import sys
    sys.path.insert(0, 'model')
    return HookedTransformer, HookedTransformerConfig, np, plt, torch


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load Original Model & Create HookedTransformer

    We'll create a TransformerLens model with matching architecture and
    transfer the trained weights.
    """)
    return


@app.cell
def _(torch):
    # Token definitions
    EOS_TOKEN = 11
    VOCAB_SIZE = 12

    # Load original checkpoint
    checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
    config = checkpoint['config']
    original_state_dict = checkpoint['model_state_dict']

    print(f"Original model config: {config}")
    print(f"Trained for {checkpoint['step']} steps, accuracy: {checkpoint['accuracy']:.2%}")
    return EOS_TOKEN, config, original_state_dict


@app.cell
def _(HookedTransformerConfig, config):
    # Create HookedTransformer config matching our model
    tl_config = HookedTransformerConfig(
        n_layers=1,
        d_model=config['model_dim'],  # 64
        n_heads=config['num_heads'],  # 4
        d_head=config['model_dim'] // config['num_heads'],  # 16
        d_mlp=config['model_dim'] * 4,  # 256 (FFN intermediate)
        d_vocab=12,  # 0-9 + PAD + EOS
        n_ctx=128,  # max sequence length
        act_fn="silu",  # matches our model
        normalization_type="RMS",  # RMSNorm
        attn_only=False,
        use_attn_result=True,
        use_split_qkv_input=True,
    )
    print(f"TransformerLens config created:")
    print(f"  d_model={tl_config.d_model}, n_heads={tl_config.n_heads}, d_mlp={tl_config.d_mlp}")
    return (tl_config,)


@app.cell
def _(HookedTransformer, original_state_dict, tl_config, torch):
    # Create HookedTransformer
    tl_model = HookedTransformer(tl_config)

    # Map weights from original model to TransformerLens format
    # Original keys -> TransformerLens keys
    _weight_map = {
        'vocab.weight': 'embed.W_E',
        'pos_embed.weight': 'pos_embed.W_pos',
        'atn_norm.weight': 'blocks.0.ln1.w',
        'atn.q_proj.weight': 'blocks.0.attn.W_Q',
        'atn.k_proj.weight': 'blocks.0.attn.W_K',
        'atn.v_proj.weight': 'blocks.0.attn.W_V',
        'atn.o_proj.weight': 'blocks.0.attn.W_O',
        'atn.o_proj.bias': 'blocks.0.attn.b_O',
        'ffn_norm.weight': 'blocks.0.ln2.w',
        'ffn.up_proj.weight': 'blocks.0.mlp.W_in',
        'ffn.up_proj.bias': 'blocks.0.mlp.b_in',
        'ffn.down_proj.weight': 'blocks.0.mlp.W_out',
        'ffn.down_proj.bias': 'blocks.0.mlp.b_out',
        'out_norm.weight': 'ln_final.w',
    }

    # Load what we can directly
    _new_state = tl_model.state_dict()

    # Embedding and position embedding
    _new_state['embed.W_E'] = original_state_dict['vocab.weight']
    _new_state['pos_embed.W_pos'] = original_state_dict['pos_embed.weight']

    # Layer norms (RMSNorm has only weight, no bias)
    _new_state['blocks.0.ln1.w'] = original_state_dict['atn_norm.weight']
    _new_state['blocks.0.ln2.w'] = original_state_dict['ffn_norm.weight']
    _new_state['ln_final.w'] = original_state_dict['out_norm.weight']

    # Attention weights - need to reshape for TransformerLens format
    # Original: (d_model, d_model) -> TL: (n_heads, d_model, d_head)
    _n_heads = tl_config.n_heads
    _d_head = tl_config.d_head
    _d_model = tl_config.d_model

    _W_Q = original_state_dict['atn.q_proj.weight'].T  # (d_model, d_model)
    _W_K = original_state_dict['atn.k_proj.weight'].T
    _W_V = original_state_dict['atn.v_proj.weight'].T
    _W_O = original_state_dict['atn.o_proj.weight']  # (d_model, d_model)

    _new_state['blocks.0.attn.W_Q'] = _W_Q.reshape(_d_model, _n_heads, _d_head).permute(1, 0, 2)
    _new_state['blocks.0.attn.W_K'] = _W_K.reshape(_d_model, _n_heads, _d_head).permute(1, 0, 2)
    _new_state['blocks.0.attn.W_V'] = _W_V.reshape(_d_model, _n_heads, _d_head).permute(1, 0, 2)
    _new_state['blocks.0.attn.W_O'] = _W_O.T.reshape(_n_heads, _d_head, _d_model)
    _new_state['blocks.0.attn.b_O'] = original_state_dict['atn.o_proj.bias']

    # Zero out attention biases (our model doesn't have them)
    _new_state['blocks.0.attn.b_Q'] = torch.zeros(_n_heads, _d_head)
    _new_state['blocks.0.attn.b_K'] = torch.zeros(_n_heads, _d_head)
    _new_state['blocks.0.attn.b_V'] = torch.zeros(_n_heads, _d_head)

    # MLP weights
    _new_state['blocks.0.mlp.W_in'] = original_state_dict['ffn.up_proj.weight'].T
    _new_state['blocks.0.mlp.b_in'] = original_state_dict['ffn.up_proj.bias']
    _new_state['blocks.0.mlp.W_out'] = original_state_dict['ffn.down_proj.weight'].T
    _new_state['blocks.0.mlp.b_out'] = original_state_dict['ffn.down_proj.bias']

    # Unembed (tie to embedding)
    _new_state['unembed.W_U'] = original_state_dict['vocab.weight'].T

    tl_model.load_state_dict(_new_state, strict=False)
    tl_model.eval()
    print("Weights loaded into TransformerLens model!")
    return (tl_model,)


@app.cell
def _(EOS_TOKEN, tl_model, torch):
    # Verify the model works - test on a simple input
    _test_input = torch.tensor([[3, 5, EOS_TOKEN]])  # 53 + 7 = 60
    _logits = tl_model(_test_input)
    _pred = _logits[0, -1].argmax().item()
    print(f"Test: Input [3, 5, EOS], predicted next token: {_pred}")
    print(f"Expected: 0 (since 53 + 7 = 60, ones digit is 0)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Use TransformerLens Caching for Activations

    TransformerLens makes it easy to cache all intermediate activations.
    We'll use `run_with_cache` to capture MLP activations.
    """)
    return


@app.cell
def _(EOS_TOKEN, np, tl_model, torch):
    # Collect activations using TransformerLens caching
    _n_samples = 100

    # Store activations by input digit
    mlp_post_activations = {_d: [] for _d in range(10)}  # post-activation (after SiLU)
    mlp_pre_activations = {_d: [] for _d in range(10)}   # pre-activation (before SiLU)

    with torch.no_grad():
        for _digit in range(10):
            for _ in range(_n_samples):
                _seq = torch.tensor([[_digit, 0, EOS_TOKEN]])

                # Run with cache - this captures all activations!
                _, cache = tl_model.run_with_cache(_seq)

                # Get MLP activations at EOS position (where we predict output)
                # 'blocks.0.mlp.hook_post' is after SiLU activation
                # 'blocks.0.mlp.hook_pre' is before SiLU activation
                _post = cache['blocks.0.mlp.hook_post'][0, 2, :].cpu().numpy()  # position 2 = EOS
                _pre = cache['blocks.0.mlp.hook_pre'][0, 2, :].cpu().numpy()

                mlp_post_activations[_digit].append(_post.copy())
                mlp_pre_activations[_digit].append(_pre.copy())

    # Convert to arrays
    for _d in range(10):
        mlp_post_activations[_d] = np.stack(mlp_post_activations[_d])
        mlp_pre_activations[_d] = np.stack(mlp_pre_activations[_d])

    print(f"Collected activations for {_n_samples} samples per digit")
    print(f"MLP post activation shape: {mlp_post_activations[0].shape}")
    return (mlp_post_activations,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Analyze Overflow vs Non-Overflow Patterns

    - **No overflow (0, 1, 2)**: 0+7=7, 1+7=8, 2+7=9
    - **Overflow (3-9)**: 3+7=10, 4+7=11, ..., 9+7=16
    """)
    return


@app.cell
def _(mlp_post_activations, np):
    # Build heatmap from post-SiLU activations
    _num_neurons = mlp_post_activations[0].shape[1]
    heatmap = np.zeros((10, _num_neurons))

    for _d in range(10):
        heatmap[_d, :] = mlp_post_activations[_d].mean(axis=0)

    # Define groups
    no_overflow = [0, 1, 2]
    overflow = [3, 4, 5, 6, 7, 8, 9]

    print(f"Heatmap shape: {heatmap.shape}")
    print(f"No overflow: {no_overflow} -> {[d+7 for d in no_overflow]}")
    print(f"Overflow: {overflow} -> {[(d+7)%10 for d in overflow]} (with carry)")
    return heatmap, no_overflow, overflow


@app.cell
def _(heatmap, np, plt):
    # Heatmap visualization
    _fig, _ax = plt.subplots(figsize=(14, 6))
    _im = _ax.imshow(heatmap, aspect='auto', cmap='RdBu_r',
                     vmin=-np.abs(heatmap).max(), vmax=np.abs(heatmap).max())
    _ax.set_xlabel('Neuron Index')
    _ax.set_ylabel('Input Digit')
    _ax.set_yticks(range(10))
    _ax.set_yticklabels([f'{d} → {(d+7) if d < 3 else str((d+7)%10)+"↑"}' for d in range(10)])
    _ax.set_title('MLP Post-Activation at Output Position (TransformerLens)')
    plt.colorbar(_im, ax=_ax, label='Mean Activation')
    _ax.axhline(y=2.5, color='black', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(heatmap, np, plt):
    # Cosine similarity matrix
    def _cosine(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    _sim = np.zeros((10, 10))
    for _i in range(10):
        for _j in range(10):
            _sim[_i, _j] = _cosine(heatmap[_i], heatmap[_j])

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _im = _ax.imshow(_sim, cmap='coolwarm', vmin=-1, vmax=1)
    _ax.set_xlabel('Input Digit')
    _ax.set_ylabel('Input Digit')
    _ax.set_xticks(range(10))
    _ax.set_yticks(range(10))
    _ax.set_title('Digit Similarity (Cosine) - MLP Activations')
    plt.colorbar(_im, ax=_ax, label='Cosine Similarity')

    for _i in range(10):
        for _j in range(10):
            _c = 'white' if abs(_sim[_i, _j]) > 0.5 else 'black'
            _ax.text(_j, _i, f'{_sim[_i, _j]:.2f}', ha='center', va='center', fontsize=8, color=_c)

    # Boxes around groups
    _ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor='green', linewidth=3))
    _ax.add_patch(plt.Rectangle((2.5, 2.5), 7, 7, fill=False, edgecolor='red', linewidth=3))

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(heatmap, no_overflow, np, overflow):
    # Quantify clustering
    def _cos(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    # Within-group similarities
    _no_of_sims = [_cos(heatmap[i], heatmap[j])
                   for _idx, i in enumerate(no_overflow) for j in no_overflow[_idx+1:]]
    _of_sims = [_cos(heatmap[i], heatmap[j])
                for _idx, i in enumerate(overflow) for j in overflow[_idx+1:]]
    _between = [_cos(heatmap[i], heatmap[j]) for i in no_overflow for j in overflow]

    print("=== Overflow Clustering Analysis (TransformerLens) ===")
    print(f"Within NO-OVERFLOW (0,1,2) avg similarity: {np.mean(_no_of_sims):.3f}")
    print(f"Within OVERFLOW (3-9) avg similarity: {np.mean(_of_sims):.3f}")
    print(f"BETWEEN groups avg similarity: {np.mean(_between):.3f}")
    print()
    if np.mean(_no_of_sims) > np.mean(_between) and np.mean(_of_sims) > np.mean(_between):
        print("✓ Overflow and no-overflow digits form DISTINCT clusters!")
    else:
        print("✗ No clear clustering by overflow status")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. TransformerLens Bonus: Attention Pattern Analysis

    Let's also look at how attention patterns differ for overflow vs non-overflow.
    """)
    return


@app.cell
def _(EOS_TOKEN, np, tl_model, torch):
    # Collect attention patterns
    attn_patterns = {_d: [] for _d in range(10)}

    with torch.no_grad():
        for _digit in range(10):
            for _ in range(50):
                _seq = torch.tensor([[_digit, 0, EOS_TOKEN]])
                _, _cache = tl_model.run_with_cache(_seq)

                # Attention pattern shape: (batch, n_heads, seq_len, seq_len)
                _attn = _cache['blocks.0.attn.hook_pattern'][0].cpu().numpy()  # (n_heads, seq, seq)
                attn_patterns[_digit].append(_attn.copy())

    for _d in range(10):
        attn_patterns[_d] = np.stack(attn_patterns[_d])  # (n_samples, n_heads, seq, seq)

    print(f"Attention pattern shape per digit: {attn_patterns[0].shape}")
    return (attn_patterns,)


@app.cell
def _(attn_patterns, plt):
    # Average attention at EOS position (where output is predicted)
    # Looking at what the EOS token attends to
    _fig, _axes = plt.subplots(2, 5, figsize=(15, 6))

    for _d in range(10):
        _ax = _axes[_d // 5, _d % 5]
        # Average over samples, shape: (n_heads, seq, seq)
        _avg_attn = attn_patterns[_d].mean(axis=0)
        # Get attention FROM position 2 (EOS) TO all positions, averaged over heads
        _eos_attn = _avg_attn[:, 2, :].mean(axis=0)  # (seq_len,)

        _colors = ['steelblue', 'coral', 'gray']
        _ax.bar(range(3), _eos_attn, color=_colors)
        _ax.set_xticks(range(3))
        _ax.set_xticklabels(['ones', 'tens', 'EOS'])
        _ax.set_ylim(0, 1)
        _ax.set_title(f'Digit {_d}' + (' (no carry)' if _d < 3 else ' (carry)'))
        if _d % 5 == 0:
            _ax.set_ylabel('Attention Weight')

    _fig.suptitle('Attention FROM EOS Position TO Input Positions', y=1.02)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(attn_patterns, no_overflow, np, overflow, plt):
    # Compare attention patterns: overflow vs non-overflow
    _no_of_attn = np.mean([attn_patterns[d].mean(axis=0)[:, 2, :].mean(axis=0) for d in no_overflow], axis=0)
    _of_attn = np.mean([attn_patterns[d].mean(axis=0)[:, 2, :].mean(axis=0) for d in overflow], axis=0)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _x = np.arange(3)
    _w = 0.35
    _ax.bar(_x - _w/2, _no_of_attn, _w, label='No Overflow (0,1,2)', color='green', alpha=0.7)
    _ax.bar(_x + _w/2, _of_attn, _w, label='Overflow (3-9)', color='red', alpha=0.7)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(['Ones Digit', 'Tens Digit', 'EOS'])
    _ax.set_ylabel('Attention Weight')
    _ax.set_title('Attention Pattern: Overflow vs No-Overflow')
    _ax.legend()
    _ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Neuron-Level Analysis with TransformerLens

    Find neurons that specifically activate for overflow cases.
    """)
    return


@app.cell
def _(heatmap, no_overflow, np, overflow, plt):
    # Find "overflow detector" neurons
    _no_of_mean = np.mean([heatmap[d] for d in no_overflow], axis=0)
    _of_mean = np.mean([heatmap[d] for d in overflow], axis=0)
    _diff = _of_mean - _no_of_mean

    # Top neurons
    _top_of = np.argsort(_diff)[-5:][::-1]
    _top_no_of = np.argsort(_diff)[:5]

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of top neurons
    _all_top = list(_top_of) + list(_top_no_of)
    _vals = [_diff[n] for n in _all_top]
    _colors = ['red']*5 + ['green']*5
    _axes[0].barh(range(10), _vals, color=_colors)
    _axes[0].set_yticks(range(10))
    _axes[0].set_yticklabels([f'Neuron {n}' for n in _all_top])
    _axes[0].set_xlabel('Activation Difference (Overflow - No Overflow)')
    _axes[0].set_title('Top Differentiating Neurons')
    _axes[0].axvline(x=0, color='black', linewidth=0.5)

    # Show activation of top overflow neuron across all digits
    _top_neuron = _top_of[0]
    _acts = [heatmap[d, _top_neuron] for d in range(10)]
    _colors2 = ['green' if d < 3 else 'red' for d in range(10)]
    _axes[1].bar(range(10), _acts, color=_colors2)
    _axes[1].set_xlabel('Input Digit')
    _axes[1].set_ylabel('Mean Activation')
    _axes[1].set_title(f'Neuron {_top_neuron} Activation (Top Overflow Detector)')
    _axes[1].set_xticks(range(10))

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    Using TransformerLens, we analyzed:

    1. **MLP Activations**: Captured post-SiLU activations at the output position
    2. **Overflow Clustering**: Measured whether digits 0-2 (no carry) cluster separately from 3-9 (carry)
    3. **Attention Patterns**: Examined how the model attends differently for overflow vs non-overflow
    4. **Neuron Specialization**: Identified specific neurons that detect overflow conditions

    TransformerLens makes this analysis much cleaner with its built-in caching!
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
