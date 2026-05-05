# model/

Core architecture code for the 1-layer transformer and its components.

## Files

### `components.py`

Building blocks shared across all architecture variants:

| Class | Description |
|-------|-------------|
| `FFN` | Standard feed-forward: `W_down(activation(W_up(x)))` |
| `GLU` | Gated Linear Unit: `W_down(activation(W_gate(x)) * W_up(x))` |
| `MHA` | Multi-head attention with optional causal masking |
| `MoE` | Mixture of Experts with FFN experts, top-k routing, load-balancing aux loss |
| `MoEGLU` | Mixture of Experts with GLU experts |
| `resolve_activation()` | Helper to convert string ("gelu", "silu", "relu") to `nn.Module` class |

### `model.py`

`OneLayerTransformer` — the main model used in all experiments:

```
embedding + positional encoding → attention (+ residual) → FFN/GLU/MoE/MoE-GLU (+ residual) → output
```

Key constructor parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_dim` | 64 | Embedding and hidden dimension |
| `num_heads` | 4 | Number of attention heads |
| `ffn_type` | "ffn" | One of: "ffn", "glu", "moe", "moe_glu" |
| `vocab_size` | 10 | Number of input tokens |
| `max_seq_len` | 128 | Maximum sequence length |
| `use_norm` | True | RMSNorm before attention and FFN |
| `is_causal` | True | Causal attention masking |
| `tie_embeddings` | True | Share embedding and output weights |
| `activation` | "silu" | Activation function string |
| `intermediate_dim` | None | FFN hidden dim (default: 4x model_dim, 2x for GLU) |
| `num_experts` | 4 | Number of MoE experts |
| `top_k` | 1 | Top-k expert routing |
| `dropout` | 0.0 | Residual dropout rate |

For MoE/MoE-GLU, `model._aux_loss` contains the load-balancing loss after each forward pass.
