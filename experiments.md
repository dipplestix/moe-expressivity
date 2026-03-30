# Experiment Plan for MoE Expressivity Project

Based on the literature survey, here are concrete experiments organized by hypothesis.

---

## H1: Attention Computes Selection, FFN Executes Mapping

### Experiment 1.1: Component Ablation Study
**Source**: "Attention Retrieves, MLP Memorizes" (Week 4)

**Protocol**:
1. Train baseline model to convergence
2. At inference, ablate each component separately:
   - Zero attention output (keep residual + FFN)
   - Zero FFN output (keep residual + attention)
   - Freeze attention weights (random init, locked during training)
3. Measure: exact-match accuracy, digit accuracy, per-token operation accuracy
4. Stratify all metrics by carry-chain length L ∈ {0,1,2,3,4,5}

**Expected outcome**: If H1 holds, zeroing attention should destroy position-dependent behavior (which token gets +7 vs +1 vs +0), while zeroing FFN should preserve selection but fail execution (model knows *what* to do but can't compute the digit).

### Experiment 1.2: Frozen Component Training
**Source**: "Attention Retrieves, MLP Memorizes" (Week 4)

**Protocol**:
1. Train model with frozen random attention (Q, K weights locked)
2. Train model with frozen random FFN
3. Compare final accuracy vs. full training

**Expected outcome**: Frozen-attention should still learn if FFN can compensate; frozen-FFN should fail more severely if FFN handles the core digit computation.

### Experiment 1.3: Linear Probes on Residual Stream
**Source**: "Understanding Addition in Transformers" (Week 3)

**Protocol**:
1. Extract activations at four points: (a) after embedding, (b) after attention, (c) after FFN pre-act, (d) after FFN
2. Train linear probes to predict o_t ∈ {+7, +1, +0} at each point
3. Report probe accuracy at each stage

**Expected outcome**: If H1 holds, operation label should become linearly decodable after attention (selection computed), with FFN output showing strong digit prediction but similar operation decodability.

### Experiment 1.4: Direct Logit Attribution (DLA)
**Source**: "FFNs Build Predictions in Vocabulary Space" (Week 5)

**Protocol**:
1. Decompose final logits: `logits = W_U @ (embed + attn_out + ffn_out)`
2. Compute contribution of each component to correct digit logit
3. Plot attribution by position and carry-chain length

**Caveats**: DLA is correlational; confirm with causal experiments (1.1, 1.5).

### Experiment 1.5: Activation Patching / Causal Tracing
**Source**: Patchscopes (Week 14), Causal Abstraction (Week 13)

**Protocol**:
1. Create matched pairs: same digits but different carry outcomes (e.g., 193 vs 183)
2. Patch attention output from one example into the other
3. Patch FFN output from one example into the other
4. Measure change in prediction and operation classifier accuracy

**Expected outcome**: Patching attention should transfer "selection" info; patching FFN should transfer "execution" computation.

### Experiment 1.6: Parallel Streams Analysis
**Source**: "Understanding Addition in Transformers" (Week 3)

**Protocol**:
1. Analyze if model develops separate processing streams for each digit position
2. Check if different heads attend to different positions consistently
3. Measure head specialization: entropy of attention over positions

**Expected outcome**: Identify "is-last" head, "carry-detector" head, etc.

---

## H2: GLU Gates Handle Selection More Cleanly

### Experiment 2.1: Gate Activation vs Operation Label
**Source**: "Understanding Gated Neurons" (Week 7), "Bilinear MLPs" (Week 6)

**Protocol**:
1. Train GLU variant to convergence
2. For each token, record gate activations g(x)
3. Train linear probe on gate activations to predict o_t
4. Compare probe accuracy to same probe on GELU pre-activations (baseline model)

**Expected outcome**: If H2 holds, gate probe accuracy > GELU pre-act probe accuracy.

### Experiment 2.2: Gate Sparsity Analysis
**Source**: "Bilinear MLPs enable weight-based interpretability" (Week 6)

**Protocol**:
1. Measure gate activation sparsity per operation type
2. Identify neurons with high mutual information between gate value and operation
3. Plot gate activation histograms conditioned on o_t ∈ {+7, +1, +0}

**Expected outcome**: GLU should show cleaner separation—specific gates activating for specific operations.

### Experiment 2.3: Weight-Based Circuit Analysis (GLU-specific)
**Source**: "Bilinear MLPs" (Week 6)

**Protocol**:
1. Express GLU as bilinear: y = (W_up @ x) ⊙ (W_gate @ x) @ W_down
2. Compute third-order tensor representation
3. Perform eigendecomposition to find low-rank structure
4. Identify interpretable eigenvectors corresponding to {+7, +1, +0}

**Expected outcome**: GLU weights should reveal cleaner operation-specific circuits than GELU.

### Experiment 2.4: Ablating Individual Gate Neurons
**Source**: "Understanding Gated Neurons" (Week 7)

**Protocol**:
1. Identify top-k gate neurons with highest MI to operation label
2. Ablate these neurons (set gate to 0) and measure accuracy drop
3. Compare to ablating random neurons

**Expected outcome**: Operation-correlated gates should be critical; random ablation should have less impact.

### Experiment 2.5: Enrichment vs Depletion Neuron Analysis
**Source**: "Understanding Gated Neurons" (Week 7)

**Protocol**:
1. Compute cosine similarity between input and output weights for each neuron
2. Classify neurons as enrichment (positive) or depletion (negative)
3. Analyze which type handles each operation

**Expected outcome**: Characterize functional organization of GLU neurons by operation type.

---

## H3: MoE Produces Stable Expert Role Splits

### Experiment 3.1: Routing Mutual Information
**Source**: "MoE-X" (Week 11), "A Closer Look into MoE" (Week 11)

**Protocol**:
1. Train MoE variant (E=4 experts, top-1 routing)
2. For each token, record which expert was selected
3. Compute MI(expert_id, token_role) where role ∈ {last, carry-reached, pass-through}
4. Compute MI(expert_id, o_t) directly

**Expected outcome**: High MI indicates expert specialization by role.

### Experiment 3.2: Expert Routing Histograms
**Source**: "A Closer Look into MoE" (Week 11)

**Protocol**:
1. Stratify by position (first digit, middle digits, last digit)
2. Stratify by carry-chain length L
3. Plot routing distribution for each condition
4. Measure routing consistency across examples with same role

**Expected outcome**: Stable expert assignment by role, not random or load-balanced only.

### Experiment 3.3: Expert-Specific Ablation
**Source**: MoEfication (Week 8)

**Protocol**:
1. Ablate each expert individually (zero its output)
2. Measure accuracy drop stratified by token role
3. Identify which expert handles which operation

**Expected outcome**: Expert A handles +7 (last digit) → ablating A destroys last-digit accuracy but preserves others.

### Experiment 3.4: MoEfication Comparison
**Source**: "MoEfication" (Week 8)

**Protocol**:
1. Train dense FFN model to convergence
2. Convert to MoE via MoEfication (cluster neurons into experts)
3. Compare routing patterns to from-scratch MoE training
4. Measure if pre-existing circuits map to distinct experts

**Expected outcome**: MoEfication should reveal implicit specialization already present in dense model.

### Experiment 3.5: Expert Output Norm Analysis
**Source**: "A Closer Look into MoE" (Week 11)

**Protocol**:
1. Measure output norm of each expert across different inputs
2. Correlate output norm with router selection probability
3. Check if router learns to select experts with larger output norms

**Expected outcome**: Understanding router decision-making mechanism.

### Experiment 3.6: Layer-wise Expert Diversity
**Source**: "A Closer Look into MoE" (Week 11)

**Protocol** (if extending to multi-layer):
1. Measure expert diversity (entropy of routing distribution) per layer
2. Check if early layers have more uniform routing, later layers more specialized

*Note*: Less relevant for 1-layer model, but useful if scaling up.

---

## Cross-Architecture Comparisons

### Experiment 4.1: Matched Training Comparison
**Protocol**:
1. Train all three architectures (GELU FFN, GLU, MoE) with same hyperparameters
2. Match parameter count across architectures
3. Train 5-10 seeds each
4. Compare learning curves, final accuracy, and convergence speed
5. Report mean ± 95% CI

### Experiment 4.2: Circuit Complexity Comparison
**Protocol**:
1. For each architecture, count number of "active" components for each operation
2. Measure effective rank of weight matrices involved in each operation
3. Compare circuit sparsity/modularity across architectures

**Expected outcome**: GLU/MoE should show more modular circuits than GELU.

### Experiment 4.3: Generalization Tests
**Source**: "Transformers Can Do Arithmetic" (Week 1), RASP (Week 17)

**Protocol**:
1. Train on carry-chain length L ≤ 2, test on L ≥ 3
2. Train without numbers ending in 9, test on them
3. Train on 2-digit, test on 3-digit
4. Compare generalization across architectures

**Expected outcome**: More modular architectures (MoE) may generalize better to novel carry patterns.

---

## Attention Pattern Analysis

### Experiment 5.1: Head Function Characterization
**Source**: "Successor Heads" (Week 14), Transformer Circuits (Week 12)

**Protocol**:
1. For each head, compute average attention pattern by input type
2. Identify "positional" heads (attend based on position, not content)
3. Identify "content" heads (attend based on digit values)
4. Check for "is-last" head that sharply attends to final digit or SEP

### Experiment 5.2: Carry-Chain Attention Patterns
**Protocol**:
1. For inputs with different carry-chain lengths, plot attention patterns
2. Check if any head attends rightward across 9s (detecting carry propagation)
3. Measure attention weight to positions in the 9-suffix as function of L

**Expected outcome**: Should find head(s) that detect "all-9 suffix to my right".

### Experiment 5.3: Attention Pattern Clustering
**Protocol**:
1. Collect attention patterns for many examples
2. Cluster patterns using k-means or hierarchical clustering
3. Analyze if clusters correspond to operation types

---

## Causal Verification (Critical for Strong Claims)

### Experiment 6.1: Causal Scrubbing
**Source**: Causal Scrubbing (Week 13)

**Protocol**:
1. Formalize hypothesis: "attention computes o_t, FFN maps o_t to digit"
2. Construct computational graph matching this hypothesis
3. Resample activations according to hypothesis structure
4. Measure if model behavior is preserved under resampling

**Expected outcome**: If hypothesis fully explains model, resampled model should have same accuracy.

### Experiment 6.2: Interchange Intervention
**Protocol**:
1. For two examples with same digit at position t but different operations
2. Interchange the attention output for position t
3. Check if model now performs the swapped operation

**Expected outcome**: Direct causal test of whether attention output determines operation.

---

## RASP Formalization (Theoretical Baseline)

### Experiment 7.1: Write RASP Program for Add-7
**Source**: RASP (Week 17)

**Protocol**:
1. Write RASP program implementing add-7 with carry propagation
2. Determine minimum heads/layers required
3. Compare to learned model's resource usage

**Expected outcome**: Establishes theoretical baseline for circuit complexity.

---

## Suggested Experiment Priority

**Phase 1 (Core H1 tests)**:
- 1.1 Component Ablation
- 1.3 Linear Probes
- 5.1 Head Function Characterization

**Phase 2 (GLU investigation for H2)**:
- 2.1 Gate vs Operation Label
- 2.2 Gate Sparsity
- 2.4 Gate Ablation

**Phase 3 (MoE investigation for H3)**:
- 3.1 Routing MI
- 3.2 Routing Histograms
- 3.3 Expert Ablation

**Phase 4 (Causal verification)**:
- 1.5 Activation Patching
- 6.1 Causal Scrubbing

**Phase 5 (Generalization & comparison)**:
- 4.1 Matched Training
- 4.3 Generalization Tests

---

## Metrics Checklist

For all experiments, report:
- [ ] Exact-match accuracy
- [ ] Per-digit accuracy
- [ ] Per-token operation accuracy (o_t prediction)
- [ ] Stratification by carry-chain length L
- [ ] Mean ± 95% CI over 5-10 seeds
- [ ] Statistical significance tests where appropriate

## Implementation Notes

- Use `formerlens/` hooked model for activation extraction
- Hook points available: `hook_attn_out`, `blocks.0.ffn.hook_pre_act`, `blocks.0.ffn.hook_post_act`
- For GLU: `blocks.0.ffn.hook_gate_pre`, `blocks.0.ffn.hook_gate_post`, `blocks.0.ffn.hook_up`, `blocks.0.ffn.hook_fuse`
- Consider releasing reproducibility toolkit (per "Arithmetic in Transformers Explained")
