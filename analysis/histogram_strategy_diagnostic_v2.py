"""
Tightened histogram counting strategy diagnostic.

Two cleaner probes than v1:

Probe 1 (relation signal): Linear probe on attention output.
    Train a per-position linear classifier on the residual stream AFTER
    attention but BEFORE FFN, predicting the count class. Compare to a
    probe on the raw embedding (lower bound) and on the post-FFN residual
    (upper bound).

    Interpretation:
      probe_after_attn ≈ probe_after_ffn  =>  attention output already
        contains the count   =>  pure relation-based.
      probe_after_attn ≈ probe_at_embed   =>  attention adds nothing
        useful for counting  =>  pure inventory-based.
      Anything in between   =>  hybrid; the gap (after_ffn - after_attn)
        is what FFN adds.

Probe 2 (inventory signal): W_up row structure.
    Feed each TOKEN EMBEDDING (32 of them, no position, no attention)
    directly through the FFN's first projection. Get a (T, hidden) matrix
    of activations. For each neuron, find its preferred token (argmax).
    Compute selectivity = (top1 activation) / (mean activation across all
    tokens, in absolute value). Plot the distribution of preferred tokens
    across neurons.

    Interpretation:
      Most neurons highly selective (selectivity >> 1) and tokens roughly
        uniformly represented across neurons => inventory-style detectors.
      Most neurons unselective (selectivity ~= 1) => no token-identity
        memorization at the W_up level.

Outputs printed to stdout.
"""

import sys, os
os.chdir("<PATH_TO_REPO>")
sys.path.insert(0, ".")
sys.path.insert(0, "model")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import OneLayerTransformer
from data.histogram import HistogramDataset

DEVICE = "cpu"
MODEL_KEYS = {
    "model_dim", "num_heads", "ffn_type", "dropout", "vocab_size",
    "max_seq_len", "use_norm", "is_causal", "tie_embeddings",
    "activation", "intermediate_dim", "num_experts", "top_k",
}


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    config = {k: v for k, v in ckpt["config"].items() if k in MODEL_KEYS}
    num_classes = ckpt["config"]["num_classes"]
    model = OneLayerTransformer(**config).to(DEVICE)
    model.unembed = nn.Linear(config["model_dim"], num_classes, bias=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, num_classes


def collect_residuals(model, inputs):
    """Return three residual stream snapshots:
        x_embed:  embedding + pos (before attention)
        x_post_attn: x_embed + attention_output (FFN input, before ffn norm)
        x_post_ffn:  x_post_attn + ffn_output (final, before out_norm)
    """
    B, L = inputs.shape
    pos = torch.arange(L, device=inputs.device)
    with torch.no_grad():
        x_embed = model.vocab(inputs) + model.pos_embed(pos)
        attn_out = model.atn(model.atn_norm(x_embed))
        x_post_attn = x_embed + attn_out
        ffn_out = model.ffn(model.ffn_norm(x_post_attn))
        x_post_ffn = x_post_attn + ffn_out
    return x_embed, x_post_attn, x_post_ffn


def linear_probe(features, targets, n_classes, train_frac=0.7, lr=0.1, steps=2000):
    """Train a linear classifier on (features, targets) and return test acc.
    features: (N, D) tensor. targets: (N,) long tensor. Splits in order
    (no shuffle needed since dataset already random)."""
    N, D = features.shape
    n_train = int(N * train_frac)
    Xtr, ytr = features[:n_train], targets[:n_train]
    Xte, yte = features[n_train:], targets[n_train:]

    probe = nn.Linear(D, n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for step in range(steps):
        logits = probe(Xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_acc = (probe(Xte).argmax(-1) == yte).float().mean().item()
        train_acc = (probe(Xtr).argmax(-1) == ytr).float().mean().item()
    return train_acc, test_acc


def probe1_attention_readout(model, inputs, targets, n_classes):
    """Compare per-position linear probes on residual stream at three depths."""
    x_embed, x_post_attn, x_post_ffn = collect_residuals(model, inputs)
    B, L, D = x_embed.shape
    feats_e = x_embed.reshape(B * L, D)
    feats_a = x_post_attn.reshape(B * L, D)
    feats_f = x_post_ffn.reshape(B * L, D)
    targs = targets.reshape(B * L)
    e_tr, e_te = linear_probe(feats_e, targs, n_classes)
    a_tr, a_te = linear_probe(feats_a, targs, n_classes)
    f_tr, f_te = linear_probe(feats_f, targs, n_classes)
    return {
        "embed":     (e_tr, e_te),
        "post_attn": (a_tr, a_te),
        "post_ffn":  (f_tr, f_te),
    }


def probe2_wup_token_selectivity(model, T_vocab):
    """Push only token embeddings (no position) through FFN's up_proj
    (or gate*up for GLU; per-expert for MoE) and compute neuron→token
    selectivity directly from weights."""
    ffn = model.ffn

    # Pure token embedding (T, d). Skip position to isolate token signal.
    E = model.vocab.weight  # (vocab, d)
    if E.shape[0] >= T_vocab:
        E = E[:T_vocab]

    # Apply ffn_norm to be consistent with what FFN sees (approximately).
    # We use it as a stand-in; actual FFN input is post_attn residual.
    with torch.no_grad():
        E_norm = model.ffn_norm(E)

    if hasattr(ffn, "experts"):  # MoE / MoE-GLU
        T_count, d = E_norm.shape
        per_expert = []
        for e_idx, e in enumerate(ffn.experts):
            with torch.no_grad():
                if hasattr(e, "gate_proj"):
                    h = e.activation(e.gate_proj(E_norm)) * e.up_proj(E_norm)
                else:
                    h = e.activation(e.up_proj(E_norm))
            per_expert.append(h)
        # Stack experts: combined hidden is (T, total_hidden)
        H = torch.cat(per_expert, dim=-1)  # (T, E*h_e)
    elif hasattr(ffn, "gate_proj"):  # GLU
        with torch.no_grad():
            H = ffn.activation(ffn.gate_proj(E_norm)) * ffn.up_proj(E_norm)
    else:  # plain FFN
        with torch.no_grad():
            H = ffn.activation(ffn.up_proj(E_norm))

    H_np = H.numpy()  # (T, hidden)
    abs_H = np.abs(H_np)
    # Selectivity per neuron: top1 / mean(abs across tokens)
    top1 = abs_H.max(axis=0)
    mean_abs = abs_H.mean(axis=0) + 1e-8
    selectivity = top1 / mean_abs  # (hidden,)
    preferred = abs_H.argmax(axis=0)  # which token does each neuron prefer

    n_neurons = H_np.shape[1]
    # Distribution of preferred tokens
    counts = np.bincount(preferred, minlength=T_vocab)
    coverage = (counts > 0).sum()  # how many distinct tokens are some neuron's favorite

    # Strict selectivity: how many neurons have selectivity > 5
    # (i.e., top token activates 5x harder than average across all tokens)
    very_selective = (selectivity > 5).sum()
    moderately_selective = (selectivity > 2).sum()

    return {
        "n_neurons": n_neurons,
        "T_vocab": T_vocab,
        "mean_selectivity": float(selectivity.mean()),
        "median_selectivity": float(np.median(selectivity)),
        "frac_sel_gt2": float(moderately_selective / n_neurons),
        "frac_sel_gt5": float(very_selective / n_neurons),
        "token_coverage": int(coverage),
        "neurons_per_token_min": int(counts.min()),
        "neurons_per_token_max": int(counts.max()),
        "neurons_per_token_mean": float(counts.mean()),
    }


def main():
    dataset = HistogramDataset(T=32, L=10, n_train=1000, n_test=2000, seed=42, device=DEVICE)
    inputs = dataset.test_inputs
    targets = dataset.test_targets

    print("=" * 80)
    print("Histogram counting strategy diagnostic v2 (tightened probes)")
    print("Setup: T=32, L=10, d=128, h=512, softmax dot-product, no BOS")
    print("=" * 80)

    families = [
        ("hist_ffn_s42",     "FFN (default)"),
        ("hist_glu_s42",     "GLU"),
        ("hist_moe_s42",     "MoE"),
        ("hist_moe_glu_s42", "MoE-GLU"),
        ("hist_ffn_narrow_s42", "FFN narrow (h=128)"),
    ]

    for ckpt_name, label in families:
        path = f"checkpoints/{ckpt_name}/hist_best.pt"
        if not os.path.exists(path):
            print(f"\n{label}: MISSING {path}")
            continue
        print(f"\n--- {label} ({ckpt_name}) ---")
        model, ckpt, n_classes = load_model(path)
        print(f"  test_acc (full model): {ckpt['test_acc']:.4f}")

        probes = probe1_attention_readout(model, inputs, targets, n_classes)
        print(f"  Probe 1 — linear readout of count class from residual stream:")
        print(f"    at embed only:   train={probes['embed'][0]:.3f}  test={probes['embed'][1]:.3f}")
        print(f"    after attention: train={probes['post_attn'][0]:.3f}  test={probes['post_attn'][1]:.3f}")
        print(f"    after FFN:       train={probes['post_ffn'][0]:.3f}  test={probes['post_ffn'][1]:.3f}")
        gap = probes['post_ffn'][1] - probes['post_attn'][1]
        attn_lift = probes['post_attn'][1] - probes['embed'][1]
        print(f"    attention lift over embed:  +{attn_lift:.3f}")
        print(f"    FFN lift over attention:    +{gap:.3f}   <<<< how much FFN adds")

        sel = probe2_wup_token_selectivity(model, T_vocab=32)
        print(f"  Probe 2 — W_up neuron→token selectivity (no position, no attention):")
        print(f"    n_neurons:                    {sel['n_neurons']}")
        print(f"    mean selectivity (top1/mean): {sel['mean_selectivity']:.2f}")
        print(f"    frac neurons sel > 2:         {sel['frac_sel_gt2']:.3f}")
        print(f"    frac neurons sel > 5:         {sel['frac_sel_gt5']:.3f}")
        print(f"    token coverage:               {sel['token_coverage']}/{sel['T_vocab']}  "
              f"(distinct tokens that are SOME neuron's argmax)")
        print(f"    neurons per token: min={sel['neurons_per_token_min']}, "
              f"max={sel['neurons_per_token_max']}, "
              f"mean={sel['neurons_per_token_mean']:.1f}")


if __name__ == "__main__":
    main()
