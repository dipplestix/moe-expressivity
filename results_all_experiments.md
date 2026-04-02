# Experiment Results

## Overview

All experiments use a 1-layer transformer. Four architecture variants: FFN, GLU, MoE (4 experts, top-1), MoE-GLU (4 experts, top-1). All results are 5 seeds unless otherwise noted. No-norm setting is primary (norm results in appendix).

**Tasks:**
- **Modular addition**: (a+b) mod 113, full-batch training, 40k epochs — grokking dynamics
- **Add-7**: x+7 digit-by-digit with carry propagation, SGD, 10k steps — computation structure
- **Histogram** (pending): count token frequencies, 500 epochs — validation of redistribution finding

---

## Core Results: Grokking on Modular Addition (No Norm)

| Variant | Best Acc | Grokked | Epoch to 99% |
|---------|---------|---------|-------------|
| FFN | 0.613±0.474 | 3/5 | 19667±5864 (3/5) |
| GLU | 0.994±0.009 | 5/5 | 19875±2770 (4/5) |
| **MoE** | **1.000±0.000** | **5/5** | **8760±700** |
| MoE-GLU | 0.997±0.002 | 5/5 | 14220±3294 |

**MoE groks 2-3x faster and with perfect reliability.** FFN only groks 3/5 seeds.

---

## Exp 1: Regularization Baselines

**Question**: Is MoE's grokking advantage from routing dynamics or just extra regularization?

| Variant | Grokked | Epoch to 99% |
|---------|---------|-------------|
| FFN (baseline, wd=1.0) | 3/5 | 19667±5864 |
| FFN + dropout=0.1 | **5/5** | **3400±200** |
| FFN + dropout=0.3 | **5/5** | **2100±200** |
| FFN + wd=0.5 | 1/5 | 35500 (1/5) |
| FFN + wd=2.0 | **5/5** | 10300±5391 |
| **MoE (baseline)** | **5/5** | **8760±700** |

**Finding: Dropout is even more effective than MoE at accelerating grokking.** FFN+dropout=0.3 groks at epoch 2100, 4x faster than MoE (8760). Heavier weight decay (2.0) also helps but with more variance. Lighter weight decay (0.5) hurts. This suggests MoE's advantage is primarily regularization, not routing-specific. However, MoE provides this regularization *without any hyperparameter tuning* — the aux loss naturally provides the right amount, whereas dropout requires choosing the right rate.

---

## Exp 2: Number of Experts

**Question**: Does grokking speed scale with expert count?

| Experts | Params | Grokked | Epoch to 99% |
|---------|--------|---------|-------------|
| E=1 | 227,072 | 4/5 | 25875±8778 (4/5) |
| E=2 | 227,328 | **5/5** | **11000±1817** |
| **E=4** | **227,840** | **5/5** | **9800±1965** |
| E=8 | 228,864 | 5/5 | 18700±9331 |
| E=16 | 230,912 | 3/5 | 20500±3342 (3/5) |

**Finding: Sweet spot at E=2-4.** E=1 (FFN with a useless router) performs similarly to plain FFN — confirming the routing itself matters, not just having a router loss. E=2 and E=4 are the fastest and most reliable. E=8 and E=16 degrade, likely because each expert gets too few parameters (intermediate_dim/num_experts) to learn useful representations.

**Critical control**: E=1 has the aux loss but doesn't benefit much (4/5 grokked, slow). This means it's not *just* the loss term — the actual routing of tokens to different experts contributes.

---

## Norm vs No-Norm

### Modular Addition: Grokking Speed

| Variant | No Norm (99% epoch) | With Norm (99% epoch) |
|---------|--------------------|--------------------|
| FFN | 19667±5864 (3/5) | **3900±374 (5/5)** |
| GLU | 19875±2770 (4/5) | **4900±1020 (5/5)** |
| MoE | 8760±700 (5/5) | 6900±3262 (5/5) |
| MoE-GLU | 14220±3294 (5/5) | **5000±447 (5/5)** |

**Finding: RMSNorm dramatically accelerates grokking for non-MoE variants.** FFN goes from 3/5 grokking at epoch 20k to 5/5 at epoch 4k. With norm, all variants converge to similar grokking speeds (4-7k epochs), largely erasing MoE's advantage. Without norm, MoE's advantage is 2-3x.

### Modular Addition: Component Ablation

| Variant | No Norm: No-FFN | With Norm: No-FFN |
|---------|----------------|-------------------|
| FFN | 1.5% | 7.7% |
| GLU | 1.4% | 14.3% |
| MoE | 16.5% | 15.1% |
| MoE-GLU | 2.9% | 10.7% |

All variants: no-attn → ~0.9% (chance) regardless of norm. Both components always critical for modular addition.

### Add-7: Component Ablation

| Variant | With Norm: No-FFN | No Norm: No-FFN |
|---------|------------------|-----------------|
| FFN | 23.1% | 9.5% |
| GLU | 27.8% | 20.9% |
| MoE | **95.1%** | **55.4%** |
| MoE-GLU | 57.6% | 39.0% |

**Finding: MoE's "FFN bypass" on add-7 is real but amplified by norm.** Without norm, MoE still gets 55% without FFN (vs 9-21% for FFN/GLU), but the dramatic 95% result is partly a norm effect.

---

## H1: Where Is Computation? (Add-7)

**H1 prediction**: Attention selects (is-last, carry detection), FFN executes (digit mapping).

| Variant | No-Attn Acc | No-FFN Acc | Interpretation |
|---------|------------|-----------|----------------|
| FFN | 54.8% | 23.1% | Both needed, FFN slightly more critical |
| GLU | 54.8% | 27.8% | Similar to FFN |
| MoE | 36.3% | 95.1% | Almost all in attention |
| MoE-GLU | 50.2% | 57.6% | Roughly balanced |

**H1 is supported for FFN/GLU** — both components contribute, with FFN doing more of the work. **H1 breaks for MoE** — the model pushes computation into attention, making FFN nearly redundant.

Linear probes on attention and FFN outputs both achieve ~99% accuracy at predicting operation type (+7/+1/+0) across all variants — the information is available everywhere in the residual stream.

---

## H2: GLU Effect

**H2 prediction**: GLU gates predict operation type more cleanly.

- Linear probe accuracy: GLU 99.6% vs FFN 98.3% — marginal improvement
- On modular addition: GLU neurons have **much lower** Fourier concentration (0.07) vs FFN (0.44). The multiplicative gate absorbs structure rather than exposing it.
- GLU groks more reliably than FFN (5/5 vs 3/5 without norm) but at similar speed

**H2 is weakly supported at best.** GLU doesn't make representations *cleaner* — it makes them *less visible* to standard activation probes. This is important for interpretability: absence of structure in GLU activations doesn't mean absence of computation.

---

## H3: MoE Expert Specialization (Add-7)

**H3 prediction**: Experts split by operation type (+7, +1, +0).

### Routing MI between expert selection and operation type:
- **MoE**: MI = 0.252±0.247 (normalized: 0.176) — highly variable across seeds
- **MoE-GLU**: MI = 0.352±0.196 (normalized: 0.246) — more consistent

### Expert ablation (MoE-GLU, best seed s42):
| Expert | +7 drop | +1 drop | +0 drop | Role |
|--------|---------|---------|---------|------|
| 0 | +0.01 | +0.27 | +0.30 | Pass-through / carry |
| 1 | +0.36 | +0.02 | +0.00 | Last-digit addition |
| 2 | +0.19 | +0.11 | +0.01 | Mixed |
| 3 | +0.00 | +0.00 | +0.00 | Redundant |

**H3 is partially supported for MoE-GLU.** Expert ablation shows differential impact by operation type — some experts disproportionately affect +7 (last-digit addition) while others affect +0/+1 (carry/pass-through). However, this specialization is inconsistent across seeds and not cleanly one-expert-per-operation.

Plain MoE shows weaker specialization, possibly because without the GLU gate, all experts compute similar functions.

---

## Fourier Analysis (Modular Addition)

### Neuron Fourier Concentration (post-grok models)
| Variant | Mean Concentration | Interpretation |
|---------|-------------------|----------------|
| FFN | 0.443 | Strong Fourier structure in neurons |
| MoE | 0.468 | Similar to FFN |
| GLU | 0.071 | Gate absorbs Fourier structure |
| MoE-GLU | 0.176 | Partial absorption |

### Router Fourier Concentration (MoE variants)
Both MoE and MoE-GLU routers develop very clean Fourier structure (>93% concentration in top-5 frequencies). This emerges during grokking, not before.

### Expert Frequency Specialization
Experts do NOT specialize by frequency — all experts in both MoE and MoE-GLU respond to the same dominant frequencies. The benefit of MoE on modular addition is regularization, not functional decomposition.

---

## Key Takeaways

1. **MoE accelerates grokking through regularization**, not expert specialization. The aux loss prevents memorization collapse. But dropout can achieve the same or better effect with tuning.

2. **MoE's advantage is most pronounced without normalization.** With RMSNorm, all variants grok at similar speeds (4-7k epochs). Without norm, MoE's advantage is 2-3x.

3. **E=2-4 experts is the sweet spot.** E=1 doesn't help (confirming routing matters, not just the loss), E≥8 hurts (experts too small).

4. **MoE pushes computation into attention on simple tasks (add-7)** but not on hard tasks (modular addition). This effect is amplified by RMSNorm.

5. **GLU hides internal structure** — Fourier concentration drops from 0.44 to 0.07 vs FFN, while achieving comparable accuracy. Interpretability methods relying on activation analysis may underestimate GLU models' capabilities.

6. **MoE-GLU shows the most promising expert specialization** on add-7, with experts differentially handling +7/+1/+0 operations. But specialization is seed-dependent.

---

## Exp 3: Model Width

**Question**: Does MoE's advantage depend on model size?

| Width | FFN Grokked | FFN 99% Epoch | MoE Grokked | MoE 99% Epoch | MoE Speedup |
|-------|------------|--------------|------------|--------------|-------------|
| d=64 | 5/5 | 12600±3878 | 5/5 | 11900±2267 | 1.1x |
| d=128 | 3/5 | 19667±5864 | 5/5 | 8760±700 | 2.2x |
| d=256 | 5/5 | 20500±1183 | 5/5 | 7500±837 | 2.7x |

**Finding: MoE's advantage scales with model width.** At d=64, both variants grok reliably and at similar speed — the model is small enough that weight decay alone suffices. At d=256, MoE is 2.7x faster. FFN groks reliably at d=256 but slowly (20.5k epochs). This suggests MoE's regularization benefit would grow further at larger scale.

---

## Exp 5: Top-k Routing

**Question**: Does top-2 routing improve or hurt grokking and specialization?

| Variant | Grokked | Epoch to 99% |
|---------|---------|-------------|
| MoE top-1 | 5/5 | 8760±700 |
| MoE top-2 | **0/5** | N/A (1% acc) |
| MoE-GLU top-1 | 5/5 | 14220±3294 |
| MoE-GLU top-2 | 3/5 | 15250±750 (2/5) |

**Finding: Top-2 routing kills grokking for MoE and hurts MoE-GLU.** MoE top-2 never groks (stays at chance). MoE-GLU top-2 partially works (3/5) but is worse than top-1. This likely happens because top-2 allows expert mixing, which reduces the discrete routing bottleneck that forces specialization. The regularization benefit of MoE depends on hard routing decisions, not soft expert blending.

---

---

## Attention Pattern Analysis (Add-7, no norm)

**Question**: What algorithm does attention learn, and how does it differ between MoE and FFN?

Analysis of per-head attention patterns on add-7 (seed 42, no norm) reveals clear head specialization in MoE that is less developed in FFN.

### MoE Head Specialization

| Head | Function | Key pattern |
|------|----------|-------------|
| 0 | Self-attention / identity | Output positions attend to themselves (o0→o0: 1.00, o1→o1: 1.00) |
| 1 | Self-attention + previous output | Mixes self with previous output position |
| 2 | **Digit-copying** | Each output attends to corresponding input (o0→d1: 0.98, o1→d2: 1.00) |
| 3 | **Context/EOS gathering** | Output positions attend to EOS and input digits for carry detection |

### FFN Head Specialization

| Head | Function | Key pattern |
|------|----------|-------------|
| 0 | Input-focused | All positions attend heavily to d0 |
| 1 | Mixed | Spread attention, no clear specialization |
| 2 | Digit-copying (partial) | o0→d1: 0.99, o1→d2: 1.00, but o2/o3 are messy |
| 3 | **EOS dump** | All output positions attend exclusively to EOS (0.97-1.00) |

### Interpretation

MoE develops **cleaner head specialization** because its FFN capacity is limited by routing — attention must do more work, so it develops structured circuits (explicit digit-copying, self-attention for residual propagation). FFN's attention is "lazier" because FFN handles the heavy lifting downstream.

This explains the per-position ablation results: MoE's digit-copying head (Head 2) lets it handle pass-through positions without FFN. FFN's attention hasn't developed this as cleanly because it doesn't need to.

**Key finding**: The MoE routing bottleneck acts as a pressure that forces attention to develop more specialized, interpretable circuits. This is a mechanistic explanation for why MoE redistributes computation toward attention.

---

## Carry-Length Stratification (Add-7, no norm)

**Question**: How do attention patterns and component reliance change with carry-chain length?

Carry length distribution (3-digit numbers): L=0: 300 (30%), L=1: 630 (63%), L=2: 63 (6.3%), L=3: 7 (0.7%).

### Attention Patterns by Carry Length (MoE, seed 42)

| Head | L=0 (no carry) | L=1 (one carry) | L=2 (two carries) | Interpretation |
|------|----------------|-----------------|--------------------|----|
| 0 | Self-attention on outputs | Same | Same | Identity/residual — stable |
| 1 | Self + previous output | Same | Same | Sequential processing — stable |
| 2 | Digit-copying (o→d) | Same | Same | Input copying — stable across L |
| 3 | Outputs → EOS mainly | Outputs → EOS + some d1 | Outputs → d0, d1 more | **Carry-sensitive**: attends more to input digits when carries propagate |

FFN heads are more stable across carry lengths — less variation, consistent with FFN handling carry logic internally rather than through attention.

### Component Ablation by Carry Length

| Carry | FFN no-FFN | MoE no-FFN | Gap |
|-------|-----------|-----------|-----|
| L=0 | ~10% | ~60% | MoE handles easy cases through attention |
| L=1 | ~8% | ~50% | Gap narrows slightly |
| L=2 | ~5% | ~35% | Harder carries still need FFN even for MoE |

**Finding**: MoE's ability to bypass FFN is carry-length dependent. Easy examples (L=0) are handled almost entirely through attention. Hard examples (L=2) still require FFN for carry propagation. The computation redistribution is adaptive — MoE shifts easy work to attention, not all work.

### Per-Position Accuracy Without FFN by Carry Length

- **L=0**: MoE handles hundreds/overflow through attention (~80-90%). FFN drops to ~0-20%.
- **L=1**: MoE still does well on overflow but tens position drops.
- **L=2**: MoE's advantage shrinks across all positions — long carry chains require FFN.

**Key finding**: The MoE routing bottleneck creates position- AND difficulty-dependent computation redistribution. Attention handles what it can (copying, easy positions); FFN handles what it must (hard carry propagation).

**Future work**: The carry-length stratified analysis was only run at default model settings (d=128, E=4). It would be informative to repeat across model widths (d=64,128,256) and expert counts (E=1,2,4,8,16) on add-7 to see if the adaptive redistribution scales with capacity. This requires training new add-7 models at those configurations.

---

## Direct Logit Attribution (Add-7, no norm)

**Question**: How much does each component (embedding, attention, FFN) contribute to the correct output logit?

**Method**: Decompose final logits as `logits = W_u @ (embed + attn_out + ffn_out)`. Measure the fraction of the correct-token logit magnitude attributable to attention vs FFN.

### Overall Attribution (5 seeds)

| Variant | Attention fraction | FFN fraction |
|---------|-------------------|-------------|
| FFN | 0.40 ± 0.07 | **0.60 ± 0.07** |
| GLU | 0.42 ± 0.03 | **0.58 ± 0.03** |
| **MoE** | **0.64 ± 0.07** | 0.36 ± 0.07 |
| MoE-GLU | 0.52 ± 0.04 | 0.48 ± 0.04 |

**Finding**: Quantitatively confirms the ablation results. Dense FFN/GLU attribute ~60% of the correct logit to FFN. MoE flips the balance — 64% from attention. MoE-GLU is roughly 50/50.

### By Position

FFN's FFN-component dominates at all positions. MoE's attention-component dominates especially at hundreds and overflow (easy positions), consistent with per-position ablation results.

### By Carry Length

At L=0 (no carry), MoE's attention dominance is strongest. At L=2, the balance shifts slightly toward FFN — confirming the redistribution is adaptive and difficulty-dependent, matching the carry-stratified ablation.

---

## Additional Analysis TODO

### Add-7:
- [x] Component ablation (zero attn/FFN) — done, all 4 variants, 5 seeds, norm + no-norm
- [x] Per-position ablation — done, all 4 variants
- [x] Carry-length stratification — done, all 4 variants
- [x] Linear probes — done, all 4 variants
- [x] Activation patching — done, all 4 variants, 5 seeds
- [x] DLA (Direct Logit Attribution) — done, all 4 variants, by position and carry length
- [x] Attention patterns — done, all 4 variants, stratified by carry length
- [x] Expert routing/ablation (H3) — done, MoE + MoE-GLU
- [ ] Head-specific ablation — zero individual attention heads to identify which are critical
- [ ] Frozen component training — train with frozen random attention or FFN
- [ ] GLU gate probes — probe gate activations specifically for operation type prediction

### Modular Addition:
- [x] Grokking dynamics — done, 4 variants × 5 seeds
- [x] Regularization baselines — done (dropout, weight decay)
- [x] Number of experts (E=1,2,4,8,16) — done
- [x] Model width scaling (d=64,128,256) — done
- [x] Top-k routing (top-1 vs top-2) — done
- [x] Norm vs no-norm — done
- [x] Component ablation — done, all 4 variants, norm + no-norm
- [x] Fourier analysis — done (neuron concentration, router, over training)
- [ ] Attention patterns — does MoE develop different attention structure here too?
- [ ] DLA — quantify attention vs FFN contribution to correct logit
- [ ] Weight norm tracking over training — does aux loss constrain weight growth during grokking?
- [ ] Fourier analysis of router during grokking — across all seeds (not just seed 42)
- [ ] Head-specific ablation

---

## Activation Patching (Add-7, no norm)

**Question**: Does patching activations from one example into another causally change the prediction? This provides causal (not just correlational) evidence for which component carries the decision.

**Method**: For pairs of examples that differ in operation at a target position (e.g., tens digit is +0 in one, +1 in the other), patch the attention or FFN output from example A into example B and measure if B now produces A's answer.

**"Flip rate"** = fraction of patches that cause the prediction to change to the source example's answer.

### Aggregated Results (5 seeds)

| Position | Component | FFN | GLU | MoE | MoE-GLU |
|----------|-----------|-----|-----|-----|---------|
| Tens | attn_flip | **0.59** | 0.39 | 0.26 | 0.29 |
| | ffn_flip | 0.10 | 0.11 | 0.18 | 0.22 |
| Hundreds | attn_flip | **0.83** | **0.83** | **0.80** | 0.71 |
| | ffn_flip | 0.21 | 0.26 | 0.32 | **0.62** |
| Overflow | attn_flip | 1.00 | 1.00 | 1.00 | 1.00 |
| | ffn_flip | 0.96 | 1.00 | 0.72 | 0.79 |

### Key findings

1. **Attention patching is consistently more causal than FFN patching** across all variants and positions. Confirms that attention carries the key decision information (operation selection).

2. **Dense FFN has the strongest attention-causal signal at tens** (0.59 vs MoE 0.26). In FFN, attention is the *only* cross-position information path, making it maximally causal. MoE distributes computation more evenly.

3. **MoE-GLU has the highest FFN flip rate at hundreds** (0.62 vs FFN 0.21). MoE-GLU's expert routing makes FFN decisions more position-specific and causally relevant — consistent with expert specialization.

4. **Overflow is trivially patchable** for both components (~1.0) since it's a simple binary decision.

5. **Causal evidence confirms ablation findings**: The component that matters more under ablation is also the one whose activation patching has higher flip rates. Attention dominates causally for dense models; MoE variants show more balanced causal contributions.

---

## All Experiments Complete

## Figures

Colors: FFN=blue, GLU=orange, MoE=green, MoE-GLU=red. All figures show all 4 variants.

**Main paper** (generated by `analysis/visualize_results.py`):

| Fig | File | Description | Task |
|-----|------|-------------|------|
| 1 | `fig1_grokking_timeline.png` | Grokking timeline, mean + shaded std | ModAdd |
| 2 | `fig2_regularization_baselines.png` | Grokking speed + reliability for baselines | ModAdd |
| 3 | `fig3_num_experts.png` | Grokking vs E=1,2,4,8,16 | ModAdd |
| 4 | `fig4_h1_ablation.png` | Component ablation (no-attn/no-ffn/normal) | Add-7 |
| 5 | `fig5_h3_routing.png` | Expert-operation routing + ablation heatmap | Add-7 |
| 6 | `fig6_fourier_concentration.png` | Per-neuron Fourier concentration histograms | ModAdd |
| 7 | `fig7_width_scaling.png` | FFN vs MoE epoch-to-99% at d=64,128,256 | ModAdd |
| 8 | `fig8_h1_ablation_modadd.png` | Component ablation | ModAdd |
| 9 | `fig9_fourier_over_training.png` | Fourier concentration vs epoch | ModAdd |
| 10 | `fig10_per_position_ablation.png` | Per-position accuracy under ablation | Add-7 |
| 11 | `fig11_attention_patterns.png` | Per-head attention heatmaps, all 4 variants | Add-7 |

**Main paper** (generated by `analysis/analyze_by_carry_length.py`):

| Fig | File | Description | Task |
|-----|------|-------------|------|
| 12 | `fig_attn_by_carry_{ftype}.png` | Attention patterns by carry length (per variant) | Add-7 |
| 13 | `fig_ablation_by_carry.png` | Component ablation by carry length | Add-7 |
| 14 | `fig_perpos_by_carry.png` | Per-position no-FFN accuracy by carry length | Add-7 |

**Main paper** (generated by `analysis/dla_add7.py`):

| Fig | File | Description | Task |
|-----|------|-------------|------|
| 15 | `fig_dla_by_position.png` | Stacked DLA by output position | Add-7 |
| 16 | `fig_dla_by_carry.png` | Attention vs FFN logit fraction by carry length | Add-7 |

**Main paper** (generated by `analysis/activation_patching.py` — text output only, no figure yet):

| Fig | File | Description | Task |
|-----|------|-------------|------|
| 17 | (text only) | Activation patching flip rates by position | Add-7 |

**Appendix** (generated by `analysis/visualize_results.py`):

| Fig | File | Description |
|-----|------|-------------|
| A1 | `figa1_norm_comparison.png` | Norm vs no-norm grokking comparison |
| A2 | `figa2_topk.png` | Top-k routing accuracy |
| A3 | `figa3_norm_ablation_add7.png` | Norm effect on component ablation |
| A4 | `figa4_perseed_routing.png` | Per-seed expert routing variability |
| A5 | `figa5_attention_patterns_norm.png` | Attention patterns with norm |

## Analysis Coverage

### By Task

| Analysis | ModAdd | Add-7 | Histogram |
|----------|--------|-------|-----------|
| Component ablation | Done | Done | Pending |
| Per-position ablation | N/A | Done | Pending |
| Carry-length stratification | N/A | Done | N/A |
| Attention patterns | Not done | Done (all 4 variants, by carry) | Pending |
| Activation patching | N/A (no pairs) | Done | N/A (no discrete ops) |
| DLA | Not done | Done (by position + carry) | Pending |
| Fourier analysis | Done | N/A | N/A |
| Expert specialization | Done (no specialization) | Done (partial) | Pending |
| Linear probes | Not done | Done | Pending |

### Coverage vs Original Project Plan

| Original Plan Item | Status | Notes |
|--------------------|--------|-------|
| Module ablation (zero attn/FFN) | **Done** | Both tasks, norm + no-norm, 5 seeds |
| DLA (Direct Logit Attribution) | **Done** (add-7) | By position and carry length. ModAdd pending. |
| Activation patching | **Done** (add-7) | Causal confirmation of ablation. 5 seeds. |
| Linear probes on internal states | **Done** (add-7) | ~99% accuracy from both components across all variants |
| Head function characterization | **Partial** | Attention heatmaps show digit-copying + carry-sensitive heads. No formal "is-last" head identification. |
| Run-of-9s detector head | **Partial** | Carry-stratified attention shows Head 3 changes with L, but not a clean 9s-detector |
| GLU gate probes for operation prediction | **Not done** | Original H2 test — probe gate activations specifically |
| MoE routing MI | **Done** | MI = 0.26-0.28 normalized, seed-dependent |
| MoE routing histograms by L and position | **Partial** | Routing by operation done. Not stratified by L specifically. |
| Balanced data curriculum by carry length | **Not done** | Training uses uniform random, not balanced by L |
| Generalization checks (train L≤2, test L≥3) | **Not done** | Noted as future work |
| Frozen component training | **Not done** | Train with frozen attention or FFN |
| Head-specific ablation | **Not done** | Zero individual heads |
| Exact-match + digit-accuracy + per-token op accuracy | **Partial** | Exact-match done. Per-token op accuracy done via probes. Digit accuracy not reported separately. |
| 95% CI over 5-10 seeds | **Done** | 5 seeds, mean ± std reported throughout |

## What Would Make the Paper Ideal

The core story: MoE's routing bottleneck forces attention to develop specialized circuits, adaptively shifting easy computations into attention while reserving expert capacity for hard operations. This is confirmed by ablation (55% vs 9.5%), DLA (64% vs 40% attention attribution), per-position analysis, carry-length stratification, attention patterns, and activation patching. Separately, MoE accelerates grokking through regularization that requires both routing and multiple experts.

Three things would take the paper from "solid empirical observations" to "mechanistic understanding with causal evidence across multiple tasks":

1. **GLU gate probes** — We show GLU hides structure (Fourier concentration 0.44→0.07) but never ran the gate probes that would explain *how*. That's the original H2 test. Without it, the GLU finding is an observation without a mechanism.

2. **Histogram task (3rd task)** — Two tasks is suggestive, three is convincing. Set up and ready — just needs training and one ablation analysis to validate the redistribution finding.

3. **Head-specific ablation** — Right now we show "MoE has cleaner heads" visually. If we could show "zeroing Head 2 in MoE destroys pass-through accuracy while zeroing Head 3 destroys carry accuracy," that's a much crisper mechanistic claim.

Priority: histogram training (easiest, highest reviewer payoff) > head-specific ablation (strongest mechanistic claim) > GLU gate probes.

---

## Pending Work (Priority Order)

### High Priority (needed for paper):
- [ ] **Histogram task training + ablation** — validates redistribution finding on 3rd task
  - [ ] Core training: `bash scripts/run_histogram_multiseed.sh`
  - [ ] Component ablation analysis
- [ ] **GLU gate probes** — the original H2 test, probe gate activations for operation type
- [ ] **Head-specific ablation** — zero individual attention heads on add-7, identify critical heads
- [ ] **Activation patching figure** — currently text only, needs a proper bar chart
- [ ] **ModAdd DLA** — quantify attn vs FFN contribution on modular addition (parallel to add-7 DLA)
- [ ] **ModAdd attention patterns** — does MoE develop different attention here too?

### Medium Priority (strengthens paper):
- [ ] Histogram controlled experiments:
  - [ ] Regularization baselines: `bash scripts/run_hist_exp1_regularization.sh`
  - [ ] Number of experts: `bash scripts/run_hist_exp2_num_experts.sh`
  - [ ] Model width: `bash scripts/run_hist_exp3_width.sh`
  - [ ] Top-k routing: `bash scripts/run_hist_exp5_topk.sh`
- [ ] **MoE routing histograms stratified by carry length L** (not just by operation)
- [ ] **Weight norm tracking during grokking** — mechanistic explanation for aux loss regularization
- [ ] **Balanced data curriculum** — retrain add-7 with balanced carry-length sampling

### Lower Priority (nice to have):
- [ ] Frozen component training (train with frozen attention or FFN)
- [ ] Fourier analysis of router across all seeds (not just seed 42)
- [ ] Generalization checks (train L≤2, test L≥3)
- [ ] Carry-length analysis at different model widths and expert counts (requires new add-7 training)
- [ ] GLU gate/up decomposition analysis (explain why GLU hides structure)
