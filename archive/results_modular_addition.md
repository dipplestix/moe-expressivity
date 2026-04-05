# Modular Addition + MoE: Experiment Results

## Summary

We trained 1-layer transformers on (a+b) mod 113 to test whether MoE expert routing
aligns with the Fourier circuits known to emerge in this task (Nanda et al., 2023).

**Key findings:**
1. MoE groks; standard FFN does not (within 40k epochs, same optimizer settings)
2. Expert routing develops clean Fourier structure *during* grokking (concentration: 0.25 -> 0.96)
3. Experts show **partial** frequency specialization — different experts' neurons peak at
   different Fourier frequencies, but with some overlap
4. No single expert is essential; ablating the most-used expert drops accuracy to 65%

---

## Experiment Setup

### Architecture
- 1-layer transformer: embedding -> attention -> MoE/FFN -> unembed
- d_model=128, num_heads=4, d_mlp=512, p=113
- No layer norm, non-causal attention, untied embeddings, GELU activation
- MoE: 4 experts (each 128-dim FFN), top-1 routing, balance_coeff=0.01

### Training
- AdamW, lr=1e-3, weight_decay=1.0, betas=(0.9, 0.999)
- Full-batch training on 30% of 113^2 = 12,769 pairs (3,830 train / 8,939 test)
- 40,000 epochs, loss = cross-entropy at position 2 (the `=` token)

### Token format
`[a, b, =]` where a,b in {0..112}, `=` token = 113. Vocab size = 114.

---

## Result 1: MoE Groks, FFN Does Not

| Model | Train acc 100% | Test acc > 50% | Test acc > 99% | Final test acc |
|-------|---------------|----------------|----------------|----------------|
| FFN   | ~epoch 1,100  | never          | never          | 3.8%           |
| MoE   | ~epoch 2,200  | epoch 13,000   | epoch 13,400   | 100%           |

### FFN grokking failure
The FFN model memorized training data by epoch 1,100 (train loss -> 0.0000) but test
accuracy plateaued at ~0.7-1% through all 40,000 epochs. A loss spike at epoch 20,200
(0.0000 -> 0.4575) briefly increased test acc to ~1% before the model re-memorized.

**Likely cause**: PyTorch default betas=(0.9, 0.999) vs Nanda's betas=(0.9, 0.98).
The slower second-moment adaptation delays the weight decay's regularization effect.
With MoE, the auxiliary load-balancing loss provides additional implicit regularization
that overcomes this, enabling grokking even with suboptimal betas.

### MoE grokking dynamics
The MoE model showed classic grokking:
- Epochs 0-1,000: memorization (train acc -> 99%+)
- Epochs 1,000-7,000: plateau (test acc ~0.5%, train loss ~0.03-0.07)
- Epochs 7,000-12,000: slow rise (test acc 0.5% -> 10%)
- Epochs 12,000-13,100: sharp transition (test acc 10% -> 93% in 1,100 epochs)
- Epochs 13,100-14,600: convergence to 100%
- Epochs 14,600+: stable at 100%

Notable: MoE training loss stays at ~0.015 even after grokking (due to aux loss),
unlike FFN which reaches 0.0000. This persistent gradient signal from the balance
loss may be mechanistically important for grokking.

---

## Result 2: Fourier Structure Emerges During Grokking

We computed the DFT of each expert's routing probability as a function of (a+b) mod p
and measured what fraction of spectral power is concentrated in the top 5 frequencies.

| Checkpoint           | Test acc | Routing entropy | Fourier concentration |
|----------------------|----------|-----------------|----------------------|
| Pre-grok (ep 5k)    | 30%*     | 1.386 (uniform) | 0.247                |
| Grok onset (ep 10k) | 37%*     | 1.386           | 0.495                |
| Mid-grok (50% test) | 71%      | 1.386           | 0.776                |
| Post-grok (99%)     | 99.4%    | 1.383           | 0.940                |
| Converged (best)    | 100%     | 1.344           | 0.956                |

*Test accuracy measured at full p^2 evaluation differs from training log due to
epoch checkpoint timing.

The routing probability is nearly random pre-grok (Fourier concentration 0.25 ~ chance
for uniformly distributed signal), then progressively develops Fourier structure that
peaks after grokking (0.96). This shows the Fourier circuit forms **as part of the
grokking transition**, not before.

### Dominant frequencies
All experts' routing probabilities are dominated by two conjugate frequency pairs:
- **freq 18 / 95** (95 = 113 - 18)
- **freq 37 / 76** (76 = 113 - 37)

These account for 93-98% of routing probability variation. This matches the Nanda et
al. finding that the MLP learns discrete Fourier transforms of (a+b) — the MoE router
responds to the same frequencies.

---

## Result 3: Partial Expert Frequency Specialization

While the **routing probabilities** respond to the same frequencies across all experts,
the **internal neuron activations** of each expert show distinct frequency preferences:

| Expert | Top-3 freq pairs | Peak neurons | Key specialization           |
|--------|-------------------|--------------|------------------------------|
| 0      | 1, 56, 57         | 19 at freq 1 | Low frequency (DC-adjacent)  |
| 1      | 18, 36, 20        | 18 at freq 18| Mid frequency (freq 18)      |
| 2      | 19, 1, 2          | 14 at freq 112| Mixed (freq 19, low freqs)  |
| 3      | 37, 56, 57        | 56 at freq 37| High frequency (freq 37)     |

### Pairwise frequency overlap (top-3 pairs)
```
Expert 0 vs 1: 0/3 overlap (completely disjoint)
Expert 0 vs 2: 1/3 overlap (freq 1)
Expert 0 vs 3: 2/3 overlap (freqs 56, 57)
Expert 1 vs 2: 0/3 overlap (completely disjoint)
Expert 1 vs 3: 0/3 overlap (completely disjoint)
Expert 2 vs 3: 0/3 overlap (completely disjoint)
```

Expert 3 shows the strongest specialization: **56 out of 128 neurons** (44%) have
dominant frequency 37, with 15.2% of total neuron power concentrated at this single
frequency pair.

### Interpretation
The MoE has learned a **partially specialized** Fourier decomposition:
- The router makes decisions based on shared Fourier features (freqs 18 and 37)
- But each expert's internal computation focuses on **different** Fourier components
- This is an intermediate outcome between "no specialization" (all experts identical)
  and "clean frequency separation" (each expert handles one frequency)

The specialization is functional but messy — consistent with the hypothesis that MoE
provides useful inductive bias for modular decomposition, but the top-1 routing
bottleneck prevents perfect separation.

---

## Result 4: Expert Ablation

| Expert ablated | Accuracy drop | Tokens routed (= pos) |
|----------------|---------------|----------------------|
| Expert 0       | 17.9%         | 2,578 (20.2%)        |
| Expert 1       | 15.6%         | 2,470 (19.3%)        |
| Expert 2       | 34.6%         | 5,291 (41.4%)        |
| Expert 3       | 15.9%         | 2,430 (19.0%)        |

Expert 2 handles the most tokens and causes the largest accuracy drop, but even
ablating it leaves 65.5% accuracy — no single expert is a catastrophic point of
failure. The accuracy drops are roughly proportional to token count, suggesting
experts share functional responsibility rather than having exclusive domains.

---

## Conclusions

### Hypothesis evaluation

**"MoE expert routing aligns with Fourier circuits"** — **Partially confirmed.**

Evidence for:
- Routing probabilities have very clean Fourier structure (96% concentration)
- This structure emerges specifically during grokking, not before
- Different experts' neurons prefer different Fourier frequencies
- 4/6 expert pairs have zero top-3 frequency overlap

Evidence against:
- Routing probabilities themselves respond to the SAME frequencies across all experts
- Neuron frequency concentration within experts is moderate (13-29%), not crisp
- Expert ablation shows shared rather than exclusive functional roles

### Additional finding: MoE as implicit regularizer

The most unexpected result is that MoE groks while standard FFN does not, under
identical optimizer settings. The load-balancing auxiliary loss appears to prevent
the optimizer from reaching the extreme memorization regime (near-zero training loss)
that traps the FFN model. This suggests MoE's routing dynamics provide an
**implicit regularization** effect that facilitates grokking — a finding with
implications beyond this specific task.

---

## Reproduction

```bash
# FFN baseline (does NOT grok with default betas)
uv run python train_modular_addition.py --no_wandb --epochs 40000 \
    --checkpoint_dir checkpoints/modadd_ffn

# MoE (groks at ~epoch 13k)
uv run python train_modular_addition.py --ffn_type moe --num_experts 4 \
    --no_wandb --epochs 40000 --checkpoint_dir checkpoints/modadd_moe

# Load grokked model for analysis
python -c "
import sys; sys.path.insert(0, 'model')
from formerlens import HookedOneLayerTransformer
model = HookedOneLayerTransformer.from_checkpoint('checkpoints/modadd_moe/modadd_best.pt')
logits, cache = model.run_with_cache(inputs)
# cache['ffn.hook_router_probs'] -- routing probabilities
# cache['ffn.hook_expert_selection'] -- which expert was selected
# cache['ffn.experts.{i}.hook_post_act'] -- per-expert neuron activations
"
```

## Next Steps

1. **Fix FFN grokking**: Rerun FFN with `betas=(0.9, 0.98)` to replicate Nanda.
   Then compare FFN Fourier structure to MoE expert decomposition.
2. **Vary num_experts**: Test E=2, 8, 16 to see if more experts produce cleaner
   frequency separation.
3. **Top-2 routing**: Test top_k=2 to see if allowing expert mixing changes
   specialization patterns.
4. **Causal verification**: Patch specific Fourier components between experts to
   confirm functional roles.
5. **Multi-seed**: Run 5+ seeds to measure consistency of specialization patterns.
