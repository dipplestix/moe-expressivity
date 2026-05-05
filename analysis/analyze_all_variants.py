"""
Fourier analysis of all 4 architecture variants on modular addition.

For each variant:
1. Neuron activation Fourier concentration (all variants)
2. Router Fourier concentration (MoE variants only)
3. Expert routing patterns (MoE variants only)
4. Grokking timeline from milestone checkpoints
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
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    config = ckpt["config"]
    model = OneLayerTransformer(**config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def get_all_inputs(p=P):
    """Generate all p^2 input pairs for full evaluation."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    inputs = torch.stack([a, b, eq], dim=1)  # (p^2, 3)
    targets = (a + b) % p
    return inputs, targets, a, b


def fourier_concentration(signal, p, top_k=5):
    """Fraction of DFT power in top-k frequencies."""
    spectrum = np.abs(np.fft.fft(signal)) ** 2
    spectrum[0] = 0  # remove DC
    total = spectrum.sum()
    if total < 1e-12:
        return 0.0
    top_power = np.sort(spectrum)[-top_k:].sum()
    return float(top_power / total)


def neuron_freq_analysis(activations, a_vals, b_vals, p):
    """
    Analyze Fourier structure of neuron activations.
    activations: (p^2, num_neurons)
    Returns per-neuron dominant frequency and concentration.
    """
    num_neurons = activations.shape[1]
    ab_sum = (a_vals + b_vals) % p

    # Average activation as function of (a+b) mod p
    neuron_by_sum = np.zeros((p, num_neurons))
    for s in range(p):
        mask = ab_sum == s
        if mask.sum() > 0:
            neuron_by_sum[s] = activations[mask].mean(axis=0)

    dominant_freqs = []
    concentrations = []
    for n in range(num_neurons):
        spectrum = np.abs(np.fft.fft(neuron_by_sum[:, n])) ** 2
        spectrum[0] = 0  # remove DC
        total = spectrum.sum()
        if total < 1e-12:
            dominant_freqs.append(0)
            concentrations.append(0.0)
        else:
            dominant_freqs.append(int(np.argmax(spectrum)))
            concentrations.append(float(np.max(spectrum) / total))

    return np.array(dominant_freqs), np.array(concentrations), neuron_by_sum


def analyze_variant(name, ckpt_dir):
    """Run full analysis for one variant."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Load best model
    model, ckpt = load_model(f"{ckpt_dir}/modadd_best.pt")
    print(f"Best test acc: {ckpt['test_acc']:.4f} (epoch {ckpt['epoch']})")

    # Get all inputs
    inputs, targets, a_vals, b_vals = get_all_inputs()
    a_np = a_vals.numpy()
    b_np = b_vals.numpy()

    # Forward pass with hooks to capture activations
    ffn_type = ckpt["config"]["ffn_type"]
    is_moe = ffn_type in ("moe", "moe_glu")

    # Hook storage
    hook_data = {}

    def make_hook(key):
        def hook_fn(module, input, output):
            hook_data[key] = output.detach()
        return hook_fn

    def make_input_hook(key):
        def hook_fn(module, input, output):
            hook_data[key] = input[0].detach()
        return hook_fn

    # Register hooks
    hooks = []
    if is_moe:
        hooks.append(model.ffn.router.register_forward_hook(make_hook("router_out")))
        for i, expert in enumerate(model.ffn.experts):
            if hasattr(expert, "up_proj"):
                # FFN expert
                hooks.append(expert.up_proj.register_forward_hook(make_hook(f"expert_{i}_pre")))
            elif hasattr(expert, "gate_proj"):
                # GLU expert
                hooks.append(expert.gate_proj.register_forward_hook(make_hook(f"expert_{i}_gate")))
                hooks.append(expert.up_proj.register_forward_hook(make_hook(f"expert_{i}_up")))
    else:
        if hasattr(model.ffn, "up_proj"):
            hooks.append(model.ffn.up_proj.register_forward_hook(make_hook("ffn_pre")))
        if hasattr(model.ffn, "gate_proj"):
            hooks.append(model.ffn.gate_proj.register_forward_hook(make_hook("ffn_gate")))

    # Forward pass
    with torch.no_grad():
        logits = model(inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Accuracy check
    preds = logits[:, 2, :].argmax(dim=-1)
    acc = (preds == targets).float().mean().item()
    print(f"Full p^2 accuracy: {acc:.4f}")

    # ---- Neuron Fourier Analysis ----
    print(f"\n--- Neuron Fourier Analysis ---")

    if is_moe:
        # For MoE: analyze router + per-expert neurons
        router_logits = hook_data["router_out"]  # (p^2*3, E) — flattened by MoE
        # Reshape back to (p^2, 3, E) then take position 2 (= token)
        num_experts = router_logits.shape[-1]
        router_logits_3d = router_logits.view(-1, 3, num_experts)
        router_probs = F.softmax(router_logits_3d[:, 2, :], dim=-1).numpy()  # position 2 (= token)
        num_experts = router_probs.shape[1]

        # Router Fourier analysis
        print(f"\nRouter Fourier concentration (top-5 freqs):")
        ab_sum = (a_np + b_np) % P
        for e in range(num_experts):
            # Average routing prob as function of (a+b) mod p
            prob_by_sum = np.zeros(P)
            for s in range(P):
                mask = ab_sum == s
                prob_by_sum[s] = router_probs[mask, e].mean()
            fc = fourier_concentration(prob_by_sum, P)
            # Get dominant frequencies
            spectrum = np.abs(np.fft.fft(prob_by_sum)) ** 2
            spectrum[0] = 0
            top_freqs = np.argsort(spectrum)[-3:][::-1]
            print(f"  Expert {e}: concentration={fc:.3f}, top freqs={list(top_freqs)}")

        # Expert selection stats
        selections = router_probs.argmax(axis=1)
        print(f"\nExpert usage (= position):")
        for e in range(num_experts):
            count = (selections == e).sum()
            print(f"  Expert {e}: {count}/{len(selections)} ({100*count/len(selections):.1f}%)")

        # Per-expert neuron analysis (for tokens routed to each expert)
        print(f"\nPer-expert neuron frequency specialization:")
        for e in range(num_experts):
            key_pre = f"expert_{e}_pre"
            key_gate = f"expert_{e}_gate"
            if key_pre in hook_data:
                # FFN expert - pre-activation after up_proj
                act = hook_data[key_pre].numpy()
                # These are only for tokens routed to this expert, but the hook
                # captures all tokens that went through. We need the full-input version.
                # Actually, MoE dispatches only masked tokens to each expert,
                # so we need a different approach. Let's do the full-model neuron analysis instead.
                pass

        # Full neuron analysis: run each expert on ALL inputs to get full activation maps
        print(f"\nFull expert activation Fourier analysis (all p^2 inputs):")
        # Get the FFN input (post-attention residual, normed)
        ffn_input_data = {}
        def capture_ffn_input(module, input, output):
            ffn_input_data["x"] = input[0].detach()
        h = model.ffn.register_forward_hook(capture_ffn_input)
        # Need to capture input, not output
        h.remove()
        h = model.ffn.register_forward_pre_hook(lambda m, inp: ffn_input_data.update({"x": inp[0].detach()}))
        with torch.no_grad():
            model(inputs)
        h.remove()

        ffn_in = ffn_input_data["x"][:, 2, :]  # (p^2, d_model) at = position

        for e in range(num_experts):
            expert = model.ffn.experts[e]
            with torch.no_grad():
                if hasattr(expert, "gate_proj"):
                    # GLU expert
                    gate = expert.activation(expert.gate_proj(ffn_in))
                    up = expert.up_proj(ffn_in)
                    act = (gate * up).numpy()
                else:
                    # FFN expert
                    act = expert.activation(expert.up_proj(ffn_in)).numpy()

            dom_freqs, concs, _ = neuron_freq_analysis(act, a_np, b_np, P)
            mean_conc = concs.mean()
            # Count neurons per dominant frequency
            freq_counts = {}
            for f in dom_freqs:
                freq_counts[f] = freq_counts.get(f, 0) + 1
            top_freq_counts = sorted(freq_counts.items(), key=lambda x: -x[1])[:3]
            print(f"  Expert {e}: mean_concentration={mean_conc:.3f}, "
                  f"top freq->count: {top_freq_counts}")

        # Pairwise frequency overlap
        print(f"\nPairwise expert frequency overlap (top-3 freq pairs):")
        expert_top_freqs = []
        for e in range(num_experts):
            expert = model.ffn.experts[e]
            with torch.no_grad():
                if hasattr(expert, "gate_proj"):
                    gate = expert.activation(expert.gate_proj(ffn_in))
                    up = expert.up_proj(ffn_in)
                    act = (gate * up).numpy()
                else:
                    act = expert.activation(expert.up_proj(ffn_in)).numpy()
            dom_freqs, concs, _ = neuron_freq_analysis(act, a_np, b_np, P)
            freq_counts = {}
            for f in dom_freqs:
                freq_counts[f] = freq_counts.get(f, 0) + 1
            top3 = set(f for f, _ in sorted(freq_counts.items(), key=lambda x: -x[1])[:3])
            expert_top_freqs.append(top3)

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                overlap = expert_top_freqs[i] & expert_top_freqs[j]
                print(f"  Expert {i} vs {j}: {len(overlap)}/3 overlap {overlap if overlap else '{}'}")

    else:
        # Non-MoE: analyze FFN neurons directly
        if "ffn_pre" in hook_data:
            # FFN variant
            act = hook_data["ffn_pre"][:, 2, :].numpy()  # (p^2, hidden) at = position
            act = np.maximum(act, 0)  # approximate post-activation for GELU
        elif "ffn_gate" in hook_data:
            # GLU variant - use gated activation
            # Re-run to get proper post-gate activation
            ffn_input_data = {}
            h = model.ffn.register_forward_pre_hook(lambda m, inp: ffn_input_data.update({"x": inp[0].detach()}))
            with torch.no_grad():
                model(inputs)
            h.remove()
            ffn_in = ffn_input_data["x"][:, 2, :]
            with torch.no_grad():
                gate = model.ffn.activation(model.ffn.gate_proj(ffn_in))
                up = model.ffn.up_proj(ffn_in)
                act = (gate * up).numpy()
        else:
            print("  No activation data captured")
            return {}

        dom_freqs, concs, _ = neuron_freq_analysis(act, a_np, b_np, P)
        mean_conc = concs.mean()
        freq_counts = {}
        for f in dom_freqs:
            freq_counts[f] = freq_counts.get(f, 0) + 1
        top_freq_counts = sorted(freq_counts.items(), key=lambda x: -x[1])[:5]
        print(f"  Mean neuron Fourier concentration: {mean_conc:.3f}")
        print(f"  Top freq -> neuron count: {top_freq_counts}")
        print(f"  Num neurons: {len(dom_freqs)}")

        # Overall FFN output Fourier concentration
        ffn_input_data = {}
        h = model.ffn.register_forward_pre_hook(lambda m, inp: ffn_input_data.update({"x": inp[0].detach()}))
        with torch.no_grad():
            model(inputs)
        h.remove()
        ffn_in = ffn_input_data["x"][:, 2, :]
        with torch.no_grad():
            ffn_out = model.ffn(ffn_input_data["x"])[:, 2, :].numpy()

        # Fourier concentration of each output dimension
        ab_sum = (a_np + b_np) % P
        output_concs = []
        for d in range(ffn_out.shape[1]):
            signal = np.zeros(P)
            for s in range(P):
                mask = ab_sum == s
                signal[s] = ffn_out[mask, d].mean()
            output_concs.append(fourier_concentration(signal, P))
        print(f"  FFN output mean Fourier concentration: {np.mean(output_concs):.3f}")

    # ---- Grokking timeline ----
    print(f"\n--- Grokking Timeline ---")
    import os
    for milestone in ["test50", "test90", "test99"]:
        path = f"{ckpt_dir}/modadd_{milestone}.pt"
        if os.path.exists(path):
            mc = torch.load(path, weights_only=False, map_location=DEVICE)
            print(f"  {milestone}: epoch {mc['epoch']}, test_acc={mc['test_acc']:.4f}")
    print(f"  best: epoch {ckpt['epoch']}, test_acc={ckpt['test_acc']:.4f}")

    return {
        "name": name,
        "test_acc": ckpt["test_acc"],
        "epoch": ckpt["epoch"],
        "ffn_type": ffn_type,
    }


if __name__ == "__main__":
    variants = [
        ("FFN", "checkpoints/modadd_ffn"),
        ("GLU", "checkpoints/modadd_glu"),
        ("MoE (FFN experts)", "checkpoints/modadd_moe"),
        ("MoE-GLU", "checkpoints/modadd_moe_glu"),
    ]

    results = []
    for name, ckpt_dir in variants:
        try:
            r = analyze_variant(name, ckpt_dir)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Test Acc':>10} {'Grok Epoch':>12}")
    print(f"{'-'*42}")
    for r in results:
        print(f"{r['name']:<20} {r['test_acc']:>10.4f} {r['epoch']:>12}")
