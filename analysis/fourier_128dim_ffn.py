"""
Compute per-neuron Fourier concentration for 128-dim dense FFN on modular addition.

Reviewer concern (W5/Q1): GLU has 128 neurons vs FFN's 256. Maybe the per-neuron
Fourier concentration drop in GLU (0.44 -> 0.07) is just due to fewer neurons,
not the multiplicative gate. This script tests that by measuring per-neuron
concentration for a 128-dim dense FFN.

If 128-dim FFN concentration is still ~0.44, the gating mechanism causes the
opacity, not the neuron count.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'model')
from model import OneLayerTransformer

P = 113
SEEDS = [42, 137, 256, 512, 1024]


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    model_keys = ['model_dim', 'num_heads', 'ffn_type', 'vocab_size', 'max_seq_len',
                  'use_norm', 'is_causal', 'tie_embeddings', 'activation', 'intermediate_dim',
                  'num_experts', 'top_k']
    model_config = {k: config[k] for k in model_keys if k in config}
    model = OneLayerTransformer(**model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def get_ffn_hidden_activations(model):
    inputs = []
    labels = []
    for a in range(P):
        for b in range(P):
            inputs.append([a, b, P])
            labels.append((a + b) % P)
    inputs = torch.tensor(inputs, dtype=torch.long)
    labels = np.array(labels)

    ffn_hidden = {}
    def hook(module, input, output):
        ffn_hidden['act'] = output.detach()
    handle = model.ffn.activation.register_forward_hook(hook)

    with torch.no_grad():
        all_hidden = []
        batch_size = 2000
        for i in range(0, len(inputs), batch_size):
            _ = model(inputs[i:i+batch_size])
            all_hidden.append(ffn_hidden['act'][:, 2, :].clone())
        hidden = torch.cat(all_hidden, dim=0).numpy()
    handle.remove()
    return hidden, labels


def per_neuron_concentration(hidden, labels):
    n_neurons = hidden.shape[1]
    concs = []
    for n in range(n_neurons):
        vals_by_class = np.zeros(P)
        counts = np.zeros(P)
        for i in range(len(labels)):
            vals_by_class[labels[i]] += hidden[i, n]
            counts[labels[i]] += 1
        vals_by_class /= np.maximum(counts, 1)
        spectrum = np.abs(np.fft.fft(vals_by_class))
        spectrum[0] = 0
        total = (spectrum**2).sum()
        if total > 0:
            concs.append((spectrum**2).max() / total)
        else:
            concs.append(0.0)
    return float(np.mean(concs)), float(np.std(concs))


def main():
    results = []
    print(f"{'Seed':<8}{'Acc':<10}{'Mean Conc':<12}{'Std Conc':<10}")
    for seed in SEEDS:
        path = f"checkpoints/modadd_ffn_128dim_s{seed}/modadd_best.pt"
        try:
            model, ckpt = load_model(path)
            acc = ckpt['test_acc']
            if acc < 0.5:
                print(f"  {seed:<6}{acc:<10.3f}SKIPPED (didn't grok)")
                continue
            hidden, labels = get_ffn_hidden_activations(model)
            mean_c, std_c = per_neuron_concentration(hidden, labels)
            results.append(mean_c)
            print(f"{seed:<8}{acc:<10.3f}{mean_c:<12.4f}{std_c:<10.4f}")
        except Exception as e:
            print(f"  {seed}: ERROR {e}")

    if results:
        arr = np.array(results)
        print(f"\nAggregate (n={len(arr)} solved seeds):")
        print(f"  Per-neuron concentration: {arr.mean():.4f} +/- {arr.std():.4f}")
        print(f"\nComparison:")
        print(f"  Dense FFN 256-dim: ~0.44")
        print(f"  Dense FFN 128-dim: {arr.mean():.4f} (THIS RESULT)")
        print(f"  GLU 128-dim:       ~0.07")
        print(f"\nVerdict: GLU opacity is due to gating, not fewer neurons.") if arr.mean() > 0.3 else print(f"\nVerdict: Fewer neurons may explain some of GLU's opacity.")


if __name__ == '__main__':
    main()
