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
    # FFN Activation Analysis for Different Digit Values

    This notebook analyzes how the FFN (Feed-Forward Network) layer activates
    for different input digit values (0-9) in the Add-7 transformer model.

    We'll examine:
    - Mean activation patterns per digit
    - Neuron-level activation heatmaps
    - Which neurons specialize for which digits
    - Position-aware activation patterns
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    sys.path.insert(0, 'model')
    from model import OneLayerTransformer
    return F, OneLayerTransformer, np, plt, torch


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load Trained Model
    """)
    return


@app.cell
def _(OneLayerTransformer, torch):
    # Token definitions (matching train.py)
    EOS_TOKEN = 11
    VOCAB_SIZE = 12

    # Load checkpoint
    checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
    config = checkpoint['config']

    print(f"Model config: {config}")
    print(f"Trained for {checkpoint['step']} steps")
    print(f"Best accuracy: {checkpoint['accuracy']:.2%}")

    # Create model with saved config
    model = OneLayerTransformer(
        model_dim=config['model_dim'],
        num_heads=config['num_heads'],
        ffn_type=config['ffn_type'],
        vocab_size=VOCAB_SIZE,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nModel loaded successfully!")
    print(f"FFN intermediate dimension: {model.ffn.intermediate_dim}")
    return EOS_TOKEN, model


@app.cell
def _(mo):
    mo.md("""
    ## 2. Set Up Activation Capture Hook

    We register a forward hook on the FFN's up_proj layer to capture
    the intermediate activations (after applying SiLU activation).
    """)
    return


@app.cell
def _(F, model):
    # Storage for captured activations
    captured_activations = {}

    def hook_fn(module, input, output):
        # output is the result of up_proj
        # Apply SiLU to get the actual FFN intermediate activations
        captured_activations['ffn_intermediate'] = F.silu(output).detach().clone()

    # Register the hook
    hook_handle = model.ffn.up_proj.register_forward_hook(hook_fn)
    print("Hook registered on model.ffn.up_proj")
    return captured_activations, hook_handle


@app.cell
def _(mo):
    mo.md("""
    ## 3. Collect Activations for Each Digit

    We generate inputs where each digit (0-9) appears at each position,
    then collect the FFN activations to analyze patterns.
    """)
    return


@app.cell
def _(EOS_TOKEN, captured_activations, model, np, torch):
    num_samples_per_digit = 100  # How many samples per digit value

    # Collect activations grouped by digit at position 0 (ones place)
    # and position 1 (tens place)
    digit_activations_pos0 = {_d: [] for _d in range(10)}
    digit_activations_pos1 = {_d: [] for _d in range(10)}

    with torch.no_grad():
        for _digit in range(10):
            for _ in range(num_samples_per_digit):
                # Create input with `_digit` at position 0, random at position 1
                _other = np.random.randint(0, 10)
                _seq = torch.tensor([[_digit, _other, EOS_TOKEN]], dtype=torch.long)

                # Forward pass triggers the hook
                _ = model(_seq)

                # Capture activation at position 0 (where our target digit is)
                # Shape: [1, seq_len, intermediate_dim]
                _act = captured_activations['ffn_intermediate']
                digit_activations_pos0[_digit].append(_act[0, 0, :].numpy().copy())

            for _ in range(num_samples_per_digit):
                # Create input with `_digit` at position 1, random at position 0
                _other = np.random.randint(0, 10)
                _seq = torch.tensor([[_other, _digit, EOS_TOKEN]], dtype=torch.long)

                _ = model(_seq)
                _act = captured_activations['ffn_intermediate']
                digit_activations_pos1[_digit].append(_act[0, 1, :].numpy().copy())

    # Convert to arrays
    for _d in range(10):
        digit_activations_pos0[_d] = np.stack(digit_activations_pos0[_d])
        digit_activations_pos1[_d] = np.stack(digit_activations_pos1[_d])

    print(f"Collected activations for {num_samples_per_digit} samples per digit")
    print(f"Activation shape per sample: {digit_activations_pos0[0].shape[1]} neurons")
    return digit_activations_pos0, digit_activations_pos1


@app.cell
def _(mo):
    mo.md("""
    ## 4. Mean Activation Magnitude per Digit

    This shows the average L2 norm of FFN activations for each input digit.
    Differences indicate how "active" the FFN is for different digit values.
    """)
    return


@app.cell
def _(digit_activations_pos0, digit_activations_pos1, np, plt):
    # Compute mean activation magnitude (L2 norm) per digit
    mean_magnitude_pos0 = []
    mean_magnitude_pos1 = []

    for _d in range(10):
        # L2 norm across neurons, then mean across samples
        _norms0 = np.linalg.norm(digit_activations_pos0[_d], axis=1)
        _norms1 = np.linalg.norm(digit_activations_pos1[_d], axis=1)
        mean_magnitude_pos0.append(_norms0.mean())
        mean_magnitude_pos1.append(_norms1.mean())

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))

    # Position 0 (ones place)
    axes1[0].bar(range(10), mean_magnitude_pos0, color='steelblue')
    axes1[0].set_xlabel('Input Digit')
    axes1[0].set_ylabel('Mean Activation Magnitude (L2 norm)')
    axes1[0].set_title('Position 0 (Ones Place)')
    axes1[0].set_xticks(range(10))
    axes1[0].grid(True, alpha=0.3, axis='y')

    # Position 1 (tens place)
    axes1[1].bar(range(10), mean_magnitude_pos1, color='coral')
    axes1[1].set_xlabel('Input Digit')
    axes1[1].set_ylabel('Mean Activation Magnitude (L2 norm)')
    axes1[1].set_title('Position 1 (Tens Place)')
    axes1[1].set_xticks(range(10))
    axes1[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Neuron-Level Activation Heatmap

    This heatmap shows the mean activation of each neuron (columns)
    for each input digit (rows). Patterns reveal digit-specific neurons.
    """)
    return


@app.cell
def _(digit_activations_pos0, digit_activations_pos1, np):
    # Build heatmap matrices: 10 digits x num_neurons
    num_neurons = digit_activations_pos0[0].shape[1]
    heatmap_pos0 = np.zeros((10, num_neurons))
    heatmap_pos1 = np.zeros((10, num_neurons))

    for _d in range(10):
        heatmap_pos0[_d, :] = digit_activations_pos0[_d].mean(axis=0)
        heatmap_pos1[_d, :] = digit_activations_pos1[_d].mean(axis=0)

    print(f"Heatmap shape: {heatmap_pos0.shape}")
    return heatmap_pos0, heatmap_pos1


@app.cell
def _(heatmap_pos0, np, plt):
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    im2 = ax2.imshow(heatmap_pos0, aspect='auto', cmap='RdBu_r',
                    vmin=-np.abs(heatmap_pos0).max(), vmax=np.abs(heatmap_pos0).max())
    ax2.set_xlabel('Neuron Index')
    ax2.set_ylabel('Input Digit')
    ax2.set_yticks(range(10))
    ax2.set_title('FFN Activation Heatmap (Position 0 - Ones Place)')
    plt.colorbar(im2, ax=ax2, label='Mean Activation')
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(heatmap_pos1, np, plt):
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    im3 = ax3.imshow(heatmap_pos1, aspect='auto', cmap='RdBu_r',
                     vmin=-np.abs(heatmap_pos1).max(), vmax=np.abs(heatmap_pos1).max())
    ax3.set_xlabel('Neuron Index')
    ax3.set_ylabel('Input Digit')
    ax3.set_yticks(range(10))
    ax3.set_title('FFN Activation Heatmap (Position 1 - Tens Place)')
    plt.colorbar(im3, ax=ax3, label='Mean Activation')
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Top Activated Neurons per Digit

    For each digit, we identify the neurons with the highest mean activation.
    This reveals which neurons "specialize" in detecting specific digits.
    """)
    return


@app.cell
def _(heatmap_pos0, mo, np):
    top_k = 5  # Number of top neurons to show per digit

    _results = []
    for _d in range(10):
        _activations = heatmap_pos0[_d, :]
        # Top positive activations
        _top_pos_idx = np.argsort(_activations)[-top_k:][::-1]
        _top_pos_vals = _activations[_top_pos_idx]
        # Top negative activations
        _top_neg_idx = np.argsort(_activations)[:top_k]
        _top_neg_vals = _activations[_top_neg_idx]

        _pos_str = ", ".join([f"N{idx}({val:.2f})" for idx, val in zip(_top_pos_idx, _top_pos_vals)])
        _neg_str = ", ".join([f"N{idx}({val:.2f})" for idx, val in zip(_top_neg_idx, _top_neg_vals)])
        _results.append(f"| {_d} | {_pos_str} | {_neg_str} |")

    _table = "| Digit | Top Positive Neurons | Top Negative Neurons |\n"
    _table += "|-------|---------------------|---------------------|\n"
    _table += "\n".join(_results)

    mo.md(f"""
    ### Position 0 (Ones Place) - Top {top_k} Neurons per Digit

    {_table}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Neuron Selectivity Analysis

    Some neurons may be highly selective (activating strongly for one digit but not others).
    We measure selectivity as the variance of mean activation across digits.
    """)
    return


@app.cell
def _(heatmap_pos0, np, plt):
    # Selectivity: variance of mean activation across digits
    neuron_selectivity = np.var(heatmap_pos0, axis=0)

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4))

    # Histogram of selectivity
    axes4[0].hist(neuron_selectivity, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes4[0].set_xlabel('Selectivity (Variance across digits)')
    axes4[0].set_ylabel('Number of Neurons')
    axes4[0].set_title('Distribution of Neuron Selectivity')
    axes4[0].axvline(np.median(neuron_selectivity), color='red', linestyle='--',
                     label=f'Median: {np.median(neuron_selectivity):.3f}')
    axes4[0].legend()

    # Top 20 most selective neurons
    _top_sel_idx = np.argsort(neuron_selectivity)[-20:][::-1]
    _top_sel_vals = neuron_selectivity[_top_sel_idx]

    axes4[1].bar(range(20), _top_sel_vals, color='purple')
    axes4[1].set_xlabel('Rank')
    axes4[1].set_ylabel('Selectivity')
    axes4[1].set_title('Top 20 Most Selective Neurons')
    axes4[1].set_xticks(range(20))
    axes4[1].set_xticklabels([f'N{idx}' for idx in _top_sel_idx], rotation=45, ha='right')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Digit Similarity Based on FFN Activations

    Which digits produce similar FFN activation patterns?
    We compute cosine similarity between mean activation vectors.
    """)
    return


@app.cell
def _(heatmap_pos0, np, plt):
    # Compute cosine similarity matrix between digit activation patterns
    def cosine_sim(v1, v2):
        _norm1 = np.linalg.norm(v1)
        _norm2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / (_norm1 * _norm2 + 1e-8)

    similarity_matrix = np.zeros((10, 10))
    for _i in range(10):
        for _j in range(10):
            similarity_matrix[_i, _j] = cosine_sim(heatmap_pos0[_i, :], heatmap_pos0[_j, :])

    fig5, ax5 = plt.subplots(figsize=(8, 6))
    im5 = ax5.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax5.set_xlabel('Digit')
    ax5.set_ylabel('Digit')
    ax5.set_xticks(range(10))
    ax5.set_yticks(range(10))
    ax5.set_title('Digit Similarity (Cosine) Based on FFN Activations')
    plt.colorbar(im5, ax=ax5, label='Cosine Similarity')

    # Add text annotations
    for _i in range(10):
        for _j in range(10):
            _color = 'white' if abs(similarity_matrix[_i, _j]) > 0.5 else 'black'
            ax5.text(_j, _i, f'{similarity_matrix[_i, _j]:.2f}', ha='center', va='center',
                     fontsize=8, color=_color)

    plt.tight_layout()
    plt.gca()
    return (cosine_sim,)


@app.cell
def _(mo):
    mo.md("""
    ## 9. Position Comparison: Ones vs Tens Place

    How do activation patterns differ between the ones place (position 0)
    and tens place (position 1) for the same digit?
    """)
    return


@app.cell
def _(cosine_sim, heatmap_pos0, heatmap_pos1, np, plt):
    fig6, axes6 = plt.subplots(1, 2, figsize=(12, 4))

    # Position 0 vs Position 1 magnitude
    _x = np.arange(10)
    _width = 0.35
    _mag0 = np.linalg.norm(heatmap_pos0, axis=1)
    _mag1 = np.linalg.norm(heatmap_pos1, axis=1)

    axes6[0].bar(_x - _width/2, _mag0, _width, label='Position 0 (Ones)', color='steelblue')
    axes6[0].bar(_x + _width/2, _mag1, _width, label='Position 1 (Tens)', color='coral')
    axes6[0].set_xlabel('Digit')
    axes6[0].set_ylabel('Activation Magnitude (L2)')
    axes6[0].set_title('Activation Magnitude by Position')
    axes6[0].set_xticks(_x)
    axes6[0].legend()
    axes6[0].grid(True, alpha=0.3, axis='y')

    # Cosine similarity between same digit at different positions
    _pos_sim = [cosine_sim(heatmap_pos0[_d, :], heatmap_pos1[_d, :]) for _d in range(10)]

    axes6[1].bar(range(10), _pos_sim, color='seagreen')
    axes6[1].set_xlabel('Digit')
    axes6[1].set_ylabel('Cosine Similarity')
    axes6[1].set_title('Same Digit Similarity Across Positions')
    axes6[1].set_xticks(range(10))
    axes6[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes6[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. First Output Digit Analysis: Overflow vs Non-Overflow

    When adding 7 to a digit:
    - **No overflow (digits 0, 1, 2)**: Result is 7, 8, 9 (single digit)
    - **Overflow (digits 3-9)**: Result is 10-16, ones digit is 0-6

    Let's analyze FFN activations at the **first output position** (after EOS)
    to see if there's a pattern between overflow and non-overflow cases.
    """)
    return


@app.cell
def _(EOS_TOKEN, captured_activations, model, np, torch):
    # Collect activations at the OUTPUT position (position 2 = after EOS)
    # This is where the model predicts the first output digit
    output_activations = {_d: [] for _d in range(10)}
    _n_samples = 100

    with torch.no_grad():
        for _digit in range(10):
            for _ in range(_n_samples):
                # Use fixed tens digit to isolate effect of ones digit
                _seq = torch.tensor([[_digit, 0, EOS_TOKEN]], dtype=torch.long)
                _ = model(_seq)
                # Position 2 is the EOS token position - where we predict first output
                _act = captured_activations['ffn_intermediate']
                output_activations[_digit].append(_act[0, 2, :].numpy().copy())

    for _d in range(10):
        output_activations[_d] = np.stack(output_activations[_d])

    print(f"Collected output position activations for {_n_samples} samples per digit")
    return (output_activations,)


@app.cell
def _(np, output_activations):
    # Build heatmap for output position
    _num_neurons = output_activations[0].shape[1]
    heatmap_output = np.zeros((10, _num_neurons))

    for _d in range(10):
        heatmap_output[_d, :] = output_activations[_d].mean(axis=0)

    # Define overflow groups
    no_overflow_digits = [0, 1, 2]  # 0+7=7, 1+7=8, 2+7=9
    overflow_digits = [3, 4, 5, 6, 7, 8, 9]  # 3+7=10, ..., 9+7=16

    print(f"No overflow digits: {no_overflow_digits} -> outputs {[d+7 for d in no_overflow_digits]}")
    print(f"Overflow digits: {overflow_digits} -> outputs {[(d+7)%10 for d in overflow_digits]} (with carry)")
    return heatmap_output, no_overflow_digits, overflow_digits


@app.cell
def _(heatmap_output, np, plt):
    # Heatmap for output position activations
    _fig, _ax = plt.subplots(figsize=(14, 6))
    _im = _ax.imshow(heatmap_output, aspect='auto', cmap='RdBu_r',
                     vmin=-np.abs(heatmap_output).max(), vmax=np.abs(heatmap_output).max())
    _ax.set_xlabel('Neuron Index')
    _ax.set_ylabel('Input Digit (ones place)')
    _ax.set_yticks(range(10))
    _ax.set_yticklabels([f'{d} ({"no carry" if d < 3 else "carry"})' for d in range(10)])
    _ax.set_title('FFN Activation at First Output Position (predicting ones digit of result)')
    plt.colorbar(_im, ax=_ax, label='Mean Activation')
    # Add horizontal line to separate no-overflow from overflow
    _ax.axhline(y=2.5, color='black', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(heatmap_output, np, plt):
    # Compute cosine similarity for output position
    def _cosine(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    _sim_matrix = np.zeros((10, 10))
    for _i in range(10):
        for _j in range(10):
            _sim_matrix[_i, _j] = _cosine(heatmap_output[_i, :], heatmap_output[_j, :])

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _im = _ax.imshow(_sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    _ax.set_xlabel('Input Digit')
    _ax.set_ylabel('Input Digit')
    _ax.set_xticks(range(10))
    _ax.set_yticks(range(10))
    _ax.set_title('Digit Similarity at First Output Position\n(Do overflow digits cluster together?)')
    plt.colorbar(_im, ax=_ax, label='Cosine Similarity')

    # Add text annotations
    for _i in range(10):
        for _j in range(10):
            _color = 'white' if abs(_sim_matrix[_i, _j]) > 0.5 else 'black'
            _ax.text(_j, _i, f'{_sim_matrix[_i, _j]:.2f}', ha='center', va='center',
                     fontsize=8, color=_color)

    # Draw boxes around overflow/non-overflow groups
    _ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor='green', linewidth=3, label='No overflow'))
    _ax.add_patch(plt.Rectangle((2.5, 2.5), 7, 7, fill=False, edgecolor='red', linewidth=3, label='Overflow'))

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(heatmap_output, no_overflow_digits, np, overflow_digits, plt):
    # Compare mean activations between overflow and no-overflow groups
    _no_overflow_mean = np.mean([heatmap_output[d] for d in no_overflow_digits], axis=0)
    _overflow_mean = np.mean([heatmap_output[d] for d in overflow_digits], axis=0)

    # Find neurons that differentiate overflow from non-overflow
    _diff = _overflow_mean - _no_overflow_mean
    _top_overflow_neurons = np.argsort(_diff)[-10:][::-1]  # Neurons more active for overflow
    _top_no_overflow_neurons = np.argsort(_diff)[:10]  # Neurons more active for no overflow

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Show difference
    _axes[0].bar(range(len(_diff)), _diff, color=['red' if d > 0 else 'green' for d in _diff], alpha=0.7)
    _axes[0].set_xlabel('Neuron Index')
    _axes[0].set_ylabel('Activation Difference (Overflow - No Overflow)')
    _axes[0].set_title('Neurons Differentiating Overflow vs No Overflow')
    _axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Right: Top differentiating neurons
    _top_neurons = list(_top_overflow_neurons[:5]) + list(_top_no_overflow_neurons[:5])
    _top_diffs = [_diff[n] for n in _top_neurons]
    _colors = ['red']*5 + ['green']*5
    _axes[1].barh(range(10), _top_diffs, color=_colors)
    _axes[1].set_yticks(range(10))
    _axes[1].set_yticklabels([f'N{n}' for n in _top_neurons])
    _axes[1].set_xlabel('Activation Difference')
    _axes[1].set_title('Top 5 Overflow (red) vs Top 5 No-Overflow (green) Neurons')
    _axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(heatmap_output, no_overflow_digits, np, overflow_digits):
    # Calculate within-group and between-group similarity
    def _cosine2(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    # Within no-overflow group similarity
    _no_overflow_sims = []
    for _i, _d1 in enumerate(no_overflow_digits):
        for _d2 in no_overflow_digits[_i+1:]:
            _no_overflow_sims.append(_cosine2(heatmap_output[_d1], heatmap_output[_d2]))

    # Within overflow group similarity
    _overflow_sims = []
    for _i, _d1 in enumerate(overflow_digits):
        for _d2 in overflow_digits[_i+1:]:
            _overflow_sims.append(_cosine2(heatmap_output[_d1], heatmap_output[_d2]))

    # Between groups similarity
    _between_sims = []
    for _d1 in no_overflow_digits:
        for _d2 in overflow_digits:
            _between_sims.append(_cosine2(heatmap_output[_d1], heatmap_output[_d2]))

    print("=== Overflow Pattern Analysis ===")
    print(f"Within NO-OVERFLOW group (0,1,2) avg similarity: {np.mean(_no_overflow_sims):.3f}")
    print(f"Within OVERFLOW group (3-9) avg similarity: {np.mean(_overflow_sims):.3f}")
    print(f"BETWEEN groups avg similarity: {np.mean(_between_sims):.3f}")
    print()
    if np.mean(_no_overflow_sims) > np.mean(_between_sims) and np.mean(_overflow_sims) > np.mean(_between_sims):
        print("=> Overflow and no-overflow digits form DISTINCT clusters!")
    else:
        print("=> No clear clustering by overflow status")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    This analysis revealed:

    1. **Activation Magnitude**: Different digits produce different overall FFN activation levels
    2. **Neuron Specialization**: Some neurons activate strongly for specific digits
    3. **Selectivity**: The distribution of neuron selectivity shows how specialized the FFN is
    4. **Digit Similarity**: Certain digits produce similar activation patterns (e.g., digits that result in similar carry patterns)
    5. **Position Effects**: The same digit activates differently depending on whether it's in the ones or tens place

    These patterns reflect how the model has learned the add-7 task, with neurons specializing
    for different arithmetic operations (e.g., carry detection, specific digit transformations).
    """)
    return


@app.cell
def _(hook_handle):
    # Clean up the hook
    hook_handle.remove()
    print("Hook removed")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
