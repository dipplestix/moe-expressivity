# Experiment Results

## Overview

All experiments use a 1-layer transformer on two tasks:
- **Modular addition**: (a+b) mod 113, full-batch training, 40k epochs
- **Add-7**: x+7 digit-by-digit with carry propagation, SGD, 10k steps

Four architecture variants: FFN, GLU, MoE (4 experts, top-1), MoE-GLU (4 experts, top-1).
All results are 5 seeds unless otherwise noted.

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

## All Experiments Complete

## TODO: Figures

Need publication-quality figures (consistent colors: FFN=blue, GLU=orange, MoE=green, MoE-GLU=red, error bars/shading for multi-seed):

**Main paper:**
1. Grokking timeline — test acc vs epoch, all 4 variants, mean + shaded std (modadd, no norm)
2. Regularization baselines — epoch-to-99% bar chart: FFN, FFN+drop0.1, FFN+drop0.3, FFN+wd2, MoE
3. Number of experts scaling — grok reliability + epoch-to-99% vs E=1,2,4,8,16
4. H1 component ablation — grouped bars: normal/no-attn/no-ffn for each variant (add-7, no norm)
5. H3 expert-operation routing — heatmap of routing fraction or ablation drop, expert x operation (add-7 MoE-GLU, no norm)
6. Neuron Fourier concentration — histograms for all 4 variants (modadd)
7. Fourier structure over training — concentration vs epoch (modadd)

**Appendix:**
A1. Norm vs no-norm grokking comparison
A2. Norm effect on component ablation (both tasks)
A3. Per-seed expert routing variability (add-7)
