"""
Diagnostic to determine whether histogram models implement
relation-based vs inventory-based counting (per Behrens et al. 2024).

Tests:
1. Attention pattern structure: relation-based attends from each position to
   other positions sharing the same token; inventory-based does not (uniform
   or input-independent attention).
2. FFN hidden activations: inventory-based has neurons that fire selectively
   for specific token identities (token-specific lookup); relation-based does
   not (since FFN only needs p=1 in that regime).
3. FFN width sensitivity: width sweep already exists (narrow=128 vs default=512
   vs d48 etc.) - if narrowing FFN below T=32 hurt sharply, supports inventory.

Outputs a printed summary; saves nothing (read-only diagnostic).
"""

import sys
import os
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


def get_attention_patterns(model, inputs):
    """Hook-free recompute of attention probabilities (model uses
    F.scaled_dot_product_attention which doesn't expose them)."""
    atn = model.atn
    B, L = inputs.shape
    with torch.no_grad():
        x = model.vocab(inputs) + model.pos_embed(torch.arange(L, device=inputs.device))
        x = model.atn_norm(x)
        q = atn.q_proj(x)
        k = atn.k_proj(x)
        q = q.view(B, L, atn.num_heads, atn.d_head).transpose(1, 2)
        k = k.view(B, L, atn.num_heads, atn.d_head).transpose(1, 2)
        scale = 1.0 / (atn.d_head ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)
        probs = F.softmax(scores, dim=-1)
    return probs  # (B, H, L, L)


def attention_relation_score(probs, inputs):
    """
    For each (batch, head, query position i), measure: what fraction of
    attention mass goes to positions j where input[j] == input[i]?

    Compare against the random-baseline expected fraction (which is
    avg group-size / L, i.e. average probability that a uniformly chosen
    position has the same token as i).

    Score = mean attention to same-token positions - baseline same-token fraction.
    Relation-based counting: score >> 0 (attention concentrates on same-token).
    Inventory-based counting: score ≈ 0 (attention does not depend on token equality).
    """
    B, H, L, _ = probs.shape
    same_token = (inputs.unsqueeze(2) == inputs.unsqueeze(1)).float()  # (B, L, L)
    same_token = same_token.unsqueeze(1)  # (B, 1, L, L) broadcast over H

    # Mass on same-token positions per (b,h,i)
    mass_same = (probs * same_token).sum(dim=-1)  # (B, H, L)

    # Baseline: fraction of positions in this sequence sharing the query token
    baseline = same_token.mean(dim=-1)  # (B, 1, L) - mean over j
    baseline = baseline.expand(-1, H, -1)

    excess = (mass_same - baseline).mean().item()
    raw = mass_same.mean().item()
    base = baseline.mean().item()
    return raw, base, excess


def attention_input_dependence(model, inputs):
    """
    Inventory-based counting can have input-INDEPENDENT attention
    (uniform mixing across positions). Compare attention patterns
    across different sequences: if they're nearly identical, attention
    isn't using token content; if they vary, attention reads tokens.
    """
    probs = get_attention_patterns(model, inputs[:128])  # (B, H, L, L)
    # Variance of attention at each (h, i, j) cell ACROSS the batch
    var_across_batch = probs.var(dim=0).mean().item()  # mean over (H, L, L)
    mean_prob = probs.mean().item()  # uniform = 1/L
    return mean_prob, var_across_batch


def ffn_token_selectivity(model, inputs, max_neurons=None):
    """
    Inventory-based counting predicts FFN neurons that selectively respond
    to specific input token identities. Run the FFN on inputs, gather
    hidden pre-activations, and compute (per-token-identity activation mean)
    selectivity for each neuron.

    Selectivity = max_t (mean activation when token == t) - mean activation
    over all tokens, normalized by the activation std.
    """
    B, L = inputs.shape
    with torch.no_grad():
        x = model.vocab(inputs) + model.pos_embed(torch.arange(L, device=inputs.device))
        x = x + model.atn(model.atn_norm(x))
        h_in = model.ffn_norm(x)  # (B, L, D)
        # Run FFN up-projection to get hidden pre-act
        ffn = model.ffn
        if hasattr(ffn, "up_proj") and hasattr(ffn, "activation") and not hasattr(ffn, "experts"):
            # Plain FFN
            pre = ffn.up_proj(h_in)
            post = ffn.activation(pre)  # (B, L, H)
        elif hasattr(ffn, "gate_proj") and hasattr(ffn, "up_proj") and not hasattr(ffn, "experts"):
            # GLU
            pre_gate = ffn.gate_proj(h_in)
            pre_up = ffn.up_proj(h_in)
            post = ffn.activation(pre_gate) * pre_up  # gated product
        elif hasattr(ffn, "experts"):
            # MoE: use full router-weighted output (combine experts)
            x_flat = h_in.view(B * L, -1)
            router_logits = ffn.router(x_flat)
            router_probs = F.softmax(router_logits, dim=-1)
            topk_w, topk_i = torch.topk(router_probs, ffn.top_k, dim=-1)
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
            # Gather only the chosen expert's hidden post-act per token (k=1 case)
            # We get per-expert hidden by running each expert and picking
            hiddens = []
            for e_idx in range(ffn.num_experts):
                e = ffn.experts[e_idx]
                if hasattr(e, "gate_proj"):
                    pe = e.activation(e.gate_proj(x_flat)) * e.up_proj(x_flat)
                else:
                    pe = e.activation(e.up_proj(x_flat))
                hiddens.append(pe)
            hiddens = torch.stack(hiddens, dim=1)  # (BL, E, H_e)
            # Pick top-1 expert per token
            chosen = hiddens.gather(1, topk_i[:, :1].unsqueeze(-1).expand(-1, 1, hiddens.shape[-1])).squeeze(1)
            post = chosen.view(B, L, -1)
        else:
            raise NotImplementedError("Unknown FFN type")

    flat_post = post.reshape(B * L, -1)  # (B*L, H)
    flat_tokens = inputs.reshape(-1)  # (B*L,)
    H = flat_post.shape[-1]
    if max_neurons is not None:
        H = min(H, max_neurons)
        flat_post = flat_post[:, :H]

    T_vocab = int(flat_tokens.max().item()) + 1
    per_token_mean = torch.zeros(T_vocab, H)
    for t in range(T_vocab):
        mask = flat_tokens == t
        if mask.any():
            per_token_mean[t] = flat_post[mask].mean(dim=0)

    overall_mean = flat_post.mean(dim=0)  # (H,)
    overall_std = flat_post.std(dim=0) + 1e-6
    # max-min spread per neuron, normalized by overall std (rough selectivity)
    spread = (per_token_mean.max(dim=0).values - per_token_mean.min(dim=0).values) / overall_std
    selective_neurons = (spread > 1.0).sum().item()
    return {
        "n_neurons": H,
        "mean_spread": float(spread.mean().item()),
        "max_spread": float(spread.max().item()),
        "n_selective": int(selective_neurons),
        "frac_selective": float(selective_neurons / H),
    }


def width_sensitivity_summary(seeds=(42, 137, 256, 512, 1024)):
    """Compare default-FFN (h=512) vs narrow-FFN (h=128) test acc.
    Both have h > T=32, so both should support inventory-based.
    Behrens predicts inventory needs p >= T; relation needs p=1.
    The 128 case still has p > T, so this isn't a clean test of the
    inventory floor - we'd need a p < T checkpoint."""
    print("\n--- FFN width sensitivity (existing checkpoints) ---")
    print(f"{'family':25s} {'h':>4} {'mean_acc':>10} {'std':>8}")
    for family, h in [("hist_ffn", 512), ("hist_ffn_narrow", 128)]:
        accs = []
        for s in seeds:
            p = f"checkpoints/{family}_s{s}/hist_best.pt"
            if not os.path.exists(p):
                continue
            ckpt = torch.load(p, weights_only=False, map_location=DEVICE)
            accs.append(ckpt["test_acc"])
        if accs:
            print(f"{family:25s} {h:>4} {np.mean(accs):>10.4f} {np.std(accs):>8.4f}  (n={len(accs)})")


def main():
    dataset = HistogramDataset(T=32, L=10, n_train=1000, n_test=2000, seed=42, device=DEVICE)
    inputs = dataset.test_inputs[:512]
    targets = dataset.test_targets[:512]

    print("=" * 70)
    print("Histogram counting strategy diagnostic (Behrens et al. 2024 framework)")
    print("Setup: T=32, L=10, d=128, h=512, softmax dot-product, no BOS")
    print("Behrens prediction for this regime: dot+sftm => inventory-based")
    print("=" * 70)

    families = [
        ("hist_ffn_s42",     "FFN (default)"),
        ("hist_glu_s42",     "GLU"),
        ("hist_moe_s42",     "MoE"),
        ("hist_moe_glu_s42", "MoE-GLU"),
        ("hist_ffn_narrow_s42", "FFN narrow (h=128)"),
        ("hist_ffn_d48_s42", "FFN d=48"),
    ]

    for ckpt_name, label in families:
        path = f"checkpoints/{ckpt_name}/hist_best.pt"
        if not os.path.exists(path):
            print(f"\n{label}: MISSING {path}")
            continue
        print(f"\n--- {label} ({ckpt_name}) ---")
        model, ckpt, _ = load_model(path)

        probs = get_attention_patterns(model, inputs)
        raw, base, excess = attention_relation_score(probs, inputs)
        mean_prob, var_across_batch = attention_input_dependence(model, inputs)

        print(f"  test_acc:                 {ckpt['test_acc']:.4f}")
        print(f"  attn mass on same-token:  {raw:.4f} (random baseline: {base:.4f})")
        print(f"  excess over baseline:     {excess:+.4f}   "
              f"<<<<  >>0 = relation-based, ~0 = inventory-based")
        print(f"  attn variance across batch: {var_across_batch:.4e} "
              f"(0 = input-independent)")

        sel = ffn_token_selectivity(model, inputs)
        print(f"  FFN neurons:              {sel['n_neurons']}")
        print(f"  mean per-neuron spread (max-min token activation / std):")
        print(f"      mean={sel['mean_spread']:.3f}  max={sel['max_spread']:.3f}")
        print(f"  fraction of neurons with spread>1 (token-selective):  "
              f"{sel['frac_selective']:.3f}  ({sel['n_selective']}/{sel['n_neurons']})")

    width_sensitivity_summary()


if __name__ == "__main__":
    main()
