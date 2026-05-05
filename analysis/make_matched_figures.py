"""Rebuild paper figures (Fig 5, Fig 10, GLU decomposition) at parameter-
matched widths. No in-figure arrows or text annotations -- all interpretation
goes in the LaTeX captions.

Functions:
  - fig10_per_position_matched(): per-position ablation on add-7 across all
        four FFN variants at matched widths (FFN h=256, GLU h=170, MoE h_E=64,
        MoE-GLU h_total=170).
  - fig5_routing_matched(): expert routing heatmap + ablation drop heatmap on
        add-7 MoE-GLU (matched width, seed 42, "best specialization" example).
  - fig_glu_decomposition_matched(): three-panel decomposition of how
        GLU's multiplicative gate destroys per-neuron Fourier structure on
        modular addition, using h=340 matched GLU.

Activation convention:
  - add-7 figures: SiLU (paper main convention; checkpoints `_d170_silu_nonorm`).
  - modadd figures: GELU (paper main convention; checkpoints `_d340`).

Outputs are written to figures/ (which is symlinked from paper/figures/).
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

# Pull shared helpers from visualize_results so we keep one source of truth
# for: the model loader, the add-7 example generator, color/label maps, and
# Fourier-concentration utilities.
from analysis.visualize_results import (
    SEEDS,
    NUM_DIGITS,
    P,
    COLORS,
    LABELS,
    load_model,
    generate_add7_examples,
)


# ---------------------------------------------------------------------------
# Fig 10: per-position ablation on add-7 (matched widths, 5 seeds).
# ---------------------------------------------------------------------------

# (variant_label_for_LABELS_dict, dir_template) -- matched widths on add-7.
ADD7_MATCHED_VARIANTS = [
    ("ffn",     "add7_ffn_nonorm_s{seed}"),
    ("glu",     "add7_glu_d170_silu_nonorm_s{seed}"),
    ("moe",     "add7_moe_nonorm_s{seed}"),
    ("moe_glu", "add7_moe_glu_d170_silu_nonorm_s{seed}"),
]


def _load_or_none(path: Path):
    if not path.exists():
        return None
    try:
        m, _ = load_model(str(path))
        return m
    except Exception as e:
        print(f"  [load-err] {path.parent.name}: {e}")
        return None


def _per_position_acc(model, x_input, targets, out_start, out_len, ablate: str | None):
    """Return per-position accuracy after ablating `attn` or `ffn` (or None for normal)."""
    if ablate == "attn":
        orig = model.atn.forward
        model.atn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    elif ablate == "ffn":
        orig = model.ffn.forward
        model.ffn.forward = lambda *a, _o=orig, **k: torch.zeros_like(_o(*a, **k))
    else:
        orig = None

    with torch.no_grad():
        preds = model(x_input).argmax(-1)
    accs = []
    for t in range(out_len):
        pos = out_start + t
        if pos < preds.shape[1]:
            accs.append((preds[:, pos] == targets[:, pos]).float().mean().item())
        else:
            accs.append(np.nan)

    if ablate == "attn":
        model.atn.forward = orig
    elif ablate == "ffn":
        model.ffn.forward = orig
    return accs


def fig10_per_position_matched():
    print("\n=== Fig 10: per-position ablation on add-7 (matched widths) ===")
    examples = generate_add7_examples()
    seqs = torch.tensor([e["seq"] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS                  # x_input position predicting output digit 0
    out_len = NUM_DIGITS + 1                # ones, tens, hundreds, overflow
    pos_labels = ["Ones\n(+7)", "Tens", "Hundreds", "Overflow"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    for ftype, tmpl in ADD7_MATCHED_VARIANTS:
        no_ffn_per_seed = []
        no_attn_per_seed = []
        for s in SEEDS:
            m = _load_or_none(ROOT / "checkpoints" / tmpl.format(seed=s) / "best_model.pt")
            if m is None:
                continue
            no_ffn_per_seed.append(_per_position_acc(m, x_input, targets, out_start, out_len, "ffn"))
            no_attn_per_seed.append(_per_position_acc(m, x_input, targets, out_start, out_len, "attn"))
        if not no_ffn_per_seed:
            print(f"  [{ftype}] NO checkpoints found for {tmpl}")
            continue
        no_ffn = np.array(no_ffn_per_seed)
        no_attn = np.array(no_attn_per_seed)
        x = np.arange(out_len)
        ax1.errorbar(x, no_ffn.mean(0), yerr=no_ffn.std(0), fmt="o-",
                     color=COLORS[ftype], label=LABELS[ftype],
                     linewidth=2.5, markersize=9, capsize=5,
                     elinewidth=1.4, capthick=1.4,
                     markeredgecolor="black", markeredgewidth=0.6)
        ax2.errorbar(x, no_attn.mean(0), yerr=no_attn.std(0), fmt="o-",
                     color=COLORS[ftype], label=LABELS[ftype],
                     linewidth=2.5, markersize=9, capsize=5,
                     elinewidth=1.4, capthick=1.4,
                     markeredgecolor="black", markeredgewidth=0.6)
        print(f"  [{ftype}] n={len(no_ffn_per_seed)} seeds")

    for ax, title in [(ax1, "(a) No FFN"),
                      (ax2, "(b) No Attention")]:
        ax.set_xticks(range(out_len))
        ax.set_xticklabels(pos_labels, fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=13)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.set_xlabel("Output Position", fontsize=15, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=15, fontweight="bold")
        ax.set_title(title, fontsize=17, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 1.0),
                     ncol=len(handles), frameon=True, fontsize=14,
                     handlelength=2.0, borderpad=0.6, columnspacing=2.5)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    fig.tight_layout()
    out = ROOT / "figures" / "fig10_per_position_ablation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out.relative_to(ROOT)}")
    paper_out = ROOT / "paper" / "images" / "add-7" / "MoE_redistributes_computation" / "fig10_per_position_ablation.png"
    paper_out.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(out, paper_out)
    print(f"  -> copied to {paper_out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Fig 5: expert routing on add-7 MoE-GLU at matched width (seed 42 example).
# ---------------------------------------------------------------------------

def fig5_routing_matched():
    print("\n=== Fig 5: expert routing on add-7 MoE-GLU (matched width, seed 42) ===")
    examples = generate_add7_examples()
    seqs = torch.tensor([e["seq"] for e in examples], dtype=torch.long)
    x_input = seqs[:, :-1]
    targets = seqs[:, 1:]
    out_start = NUM_DIGITS
    ops = ["+7", "+1", "+0"]
    op_to_idx = {op: i for i, op in enumerate(ops)}

    # Matched width MoE-GLU on add-7: total intermediate_dim = 170 across 4 experts.
    ck = ROOT / "checkpoints" / "add7_moe_glu_d170_silu_nonorm_s42" / "best_model.pt"
    model = _load_or_none(ck)
    if model is None:
        raise FileNotFoundError(ck)
    ne = model.ffn.num_experts

    # Capture router logits via forward hook.
    hook_data = {}
    h_router = model.ffn.router.register_forward_hook(
        lambda m, i, o: hook_data.update({"out": o.detach()}))
    with torch.no_grad():
        model(x_input)
    h_router.remove()

    rl = hook_data["out"].view(-1, x_input.shape[1], ne)
    rp = F.softmax(rl, dim=-1).numpy()

    routing_counts = np.zeros((3, ne))
    for i, ex in enumerate(examples):
        for t, op in enumerate(ex["ops"]):
            pos = NUM_DIGITS + t
            if pos < rp.shape[1]:
                expert = int(np.argmax(rp[i, pos]))
                routing_counts[op_to_idx[op], expert] += 1
    routing_frac = routing_counts / routing_counts.sum(axis=1, keepdims=True)

    # Per-operation ablation accuracy.
    def per_op_acc(logits):
        preds = logits.argmax(-1)
        oc = defaultdict(list)
        for i, ex in enumerate(examples):
            for t, op in enumerate(ex["ops"]):
                pos = out_start + t
                if pos < preds.shape[1]:
                    oc[op].append((preds[i, pos] == targets[i, pos]).item())
        return {op: float(np.mean(v)) for op, v in oc.items()}

    with torch.no_grad():
        normal_acc = per_op_acc(model(x_input))

    abl_matrix = np.zeros((3, ne))
    for e_idx in range(ne):
        orig = model.ffn.experts[e_idx].forward
        model.ffn.experts[e_idx].forward = lambda x, _o=orig: torch.zeros_like(_o(x))
        with torch.no_grad():
            abl_acc = per_op_acc(model(x_input))
        model.ffn.experts[e_idx].forward = orig
        for oi, op in enumerate(ops):
            abl_matrix[oi, e_idx] = normal_acc[op] - abl_acc[op]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(routing_frac, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks(range(ne)); ax1.set_xticklabels([f"Expert {i}" for i in range(ne)])
    ax1.set_yticks(range(3));  ax1.set_yticklabels(ops)
    ax1.set_title("Routing Fraction\n(MoE-GLU, matched $h_{\\text{total}}{=}170$, seed 42)")
    ax1.set_xlabel("Expert"); ax1.set_ylabel("Operation")
    plt.colorbar(im1, ax=ax1, label="Fraction of tokens")
    for i in range(3):
        for j in range(ne):
            v = routing_frac[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.5 else "black", fontsize=11)

    im2 = ax2.imshow(abl_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.8)
    ax2.set_xticks(range(ne)); ax2.set_xticklabels([f"Expert {i}" for i in range(ne)])
    ax2.set_yticks(range(3));  ax2.set_yticklabels(ops)
    ax2.set_title("Accuracy Drop When Ablated\n(MoE-GLU, matched $h_{\\text{total}}{=}170$, seed 42)")
    ax2.set_xlabel("Expert"); ax2.set_ylabel("Operation")
    plt.colorbar(im2, ax=ax2, label="Accuracy drop")
    for i in range(3):
        for j in range(ne):
            v = abl_matrix[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.4 else "black", fontsize=11)

    fig.suptitle("Expert Specialization by Operation Type (Add-7, parameter-matched, no norm)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = ROOT / "figures" / "fig5_h3_routing.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# GLU decomposition: how multiplicative gating destroys per-neuron Fourier
# structure (matched-width GLU on modular addition, seed 42).
# ---------------------------------------------------------------------------

def _neuron_fourier(activations, a_np, b_np, p):
    n_neurons = activations.shape[1]
    ab_sum = (a_np + b_np) % p
    by_sum = np.zeros((p, n_neurons))
    for s in range(p):
        mask = ab_sum == s
        if mask.sum() > 0:
            by_sum[s] = activations[mask].mean(axis=0)
    spectra = np.zeros((n_neurons, p))
    concs = np.zeros(n_neurons)
    freqs = np.zeros(n_neurons, dtype=int)
    for n in range(n_neurons):
        spec = np.abs(np.fft.fft(by_sum[:, n])) ** 2
        spec[0] = 0
        spectra[n] = spec
        total = spec.sum()
        if total > 1e-12:
            concs[n] = float(np.max(spec) / total)
            freqs[n] = int(np.argmax(spec))
    return spectra, concs, freqs


def fig_glu_decomposition_matched():
    print("\n=== GLU decomposition (matched width, modadd, seed 42) ===")
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    eq = torch.full_like(a, P)
    inputs = torch.stack([a, b, eq], dim=1)
    a_np, b_np = a.numpy(), b.numpy()

    glu_ck = ROOT / "checkpoints" / "modadd_glu_d340_s42" / "modadd_best.pt"     # matched
    ffn_ck = ROOT / "checkpoints" / "modadd_ffn_s42" / "modadd_best.pt"

    glu_model = _load_or_none(glu_ck)
    ffn_model = _load_or_none(ffn_ck)
    if glu_model is None or ffn_model is None:
        raise FileNotFoundError("Need matched-width GLU and FFN checkpoints for seed 42")

    def get_ffn_in(model):
        cap = {}
        h = model.ffn.register_forward_pre_hook(lambda m, inp: cap.update({"x": inp[0].detach()}))
        with torch.no_grad():
            model(inputs)
        h.remove()
        return cap["x"][:, 2, :]

    glu_in = get_ffn_in(glu_model)
    with torch.no_grad():
        gate = glu_model.ffn.activation(glu_model.ffn.gate_proj(glu_in)).numpy()
        up = glu_model.ffn.up_proj(glu_in).numpy()
        product = gate * up

    ffn_in = get_ffn_in(ffn_model)
    with torch.no_grad():
        ffn_act = ffn_model.ffn.activation(ffn_model.ffn.up_proj(ffn_in)).numpy()

    gate_spectra, gate_concs, gate_freqs = _neuron_fourier(gate, a_np, b_np, P)
    up_spectra,   up_concs,   up_freqs   = _neuron_fourier(up,   a_np, b_np, P)
    prod_spectra, prod_concs, prod_freqs = _neuron_fourier(product, a_np, b_np, P)
    _,            ffn_concs, _           = _neuron_fourier(ffn_act, a_np, b_np, P)

    # Pick a neuron where gate and up both have structure but the product collapses.
    mask = (gate_concs > 0.1) & (up_concs > 0.1) & (prod_concs < 0.1)
    if mask.any():
        candidates = np.flatnonzero(mask)
        n_idx = int(candidates[np.argmax(gate_concs[candidates] + up_concs[candidates])])
    else:
        n_idx = int(np.argmax(gate_concs + up_concs - prod_concs))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # Panel (a): single-neuron Fourier spectrum (gate/up/product).
    ax = axes[0]
    half = P // 2
    fold = lambda spec: spec[:half + 1]
    ax.plot(fold(gate_spectra[n_idx] / gate_spectra[n_idx].max()), color="#C44E52",
            label=f"Gate (conc={gate_concs[n_idx]:.2f})", linewidth=1.6)
    ax.plot(fold(up_spectra[n_idx] / up_spectra[n_idx].max()), color="#4878CF",
            label=f"Up (conc={up_concs[n_idx]:.2f})", linewidth=1.6)
    prod_max = max(prod_spectra[n_idx].max(), 1e-12)
    ax.plot(fold(prod_spectra[n_idx] / prod_max), color="black",
            label=f"Product (conc={prod_concs[n_idx]:.2f})", linewidth=1.6)
    ax.set_title(f"(a) Fourier Spectrum of Neuron {n_idx}")
    ax.set_xlabel("Frequency (folded)"); ax.set_ylabel("Normalized Power")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel (b): histograms of per-neuron concentration.
    ax = axes[1]
    bins = np.linspace(0, 1, 31)
    ax.hist(gate_concs, bins=bins, color="#C44E52", alpha=0.55, label="Gate", edgecolor="white", linewidth=0.5)
    ax.hist(up_concs,   bins=bins, color="#4878CF", alpha=0.55, label="Up",   edgecolor="white", linewidth=0.5)
    ax.hist(prod_concs, bins=bins, color="black",   alpha=0.65, label="Product (GLU output)",
            edgecolor="white", linewidth=0.5)
    ax.axvline(ffn_concs.mean(), color="#2ca02c", linestyle="--", linewidth=1.2,
               label=f"FFN mean = {ffn_concs.mean():.2f}")
    ax.set_title("(b) Concentration Distribution (matched GLU)")
    ax.set_xlabel("Fourier Concentration"); ax.set_ylabel("# Neurons")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel (c): summary bar chart -- means only, no in-figure arrows.
    ax = axes[2]
    labels = ["FFN\nneurons", "GLU gate", "GLU up", "GLU\ngate$\\times$up"]
    means = [ffn_concs.mean(), gate_concs.mean(), up_concs.mean(), prod_concs.mean()]
    colors = [COLORS["ffn"], "#C44E52", "#4878CF", "black"]
    bars = ax.bar(range(4), means, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Fourier Concentration")
    ax.set_ylim(0, max(means) * 1.25 + 1e-3)
    ax.set_title("(c) Gate$\\times$Up vs. FFN")
    for i, v in enumerate(means):
        ax.text(i, v + max(means) * 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("How GLU Multiplicative Gating Reshapes Per-Neuron Fourier Structure "
                 "(matched-width GLU $h{=}340$ on $(a+b)\\bmod 113$, seed 42)".replace(
                     "\\bmod", r"\,\mathrm{mod}\,"),
                 fontsize=14, y=1.04)
    fig.tight_layout()
    out = ROOT / "figures" / "fig_glu_decomposition_enhanced.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out.relative_to(ROOT)}")
    print(f"  means: FFN={ffn_concs.mean():.3f}, gate={gate_concs.mean():.3f}, "
          f"up={up_concs.mean():.3f}, product={prod_concs.mean():.3f}")


def main():
    fig10_per_position_matched()
    fig5_routing_matched()
    fig_glu_decomposition_matched()


if __name__ == "__main__":
    main()
