# formerlens/

TransformerLens-compatible hooked versions of the model and components. These expose intermediate activations via hook points for mechanistic interpretability analysis.

## Files

### `hooked_components.py`

Hooked versions of all components from `model/components.py`:

| Class | Hook points |
|-------|-------------|
| `HookedFFN` | `hook_pre_act`, `hook_post_act` |
| `HookedGLU` | `hook_gate_pre`, `hook_gate_post`, `hook_up`, `hook_fuse` |
| `HookedMHA` | `hook_q`, `hook_k`, `hook_v`, `hook_pattern`, `hook_z` |
| `HookedMoE` | `hook_router_logits`, `hook_router_probs`, `hook_expert_selection`, `hook_expert_weights`, per-expert hooks via HookedFFN, `hook_out` |
| `HookedMoEGLU` | Same as HookedMoE but with HookedGLU experts |

### `hooked_former.py`

`HookedOneLayerTransformer` — hooked version of the main model. Hook points:

- `hook_embed` — after embedding + positional encoding
- `hook_attn_pre` — attention input (after norm)
- `hook_attn_out` — attention output
- `hook_ffn_pre` — FFN input (after norm)
- `hook_logits` — final logits
- All component-level hooks from above

Usage:
```python
from formerlens import HookedOneLayerTransformer

model = HookedOneLayerTransformer.from_checkpoint('checkpoints/best_model.pt')
logits, cache = model.run_with_cache(inputs)

# Access cached activations
attn_pattern = cache['blocks.0.attn.hook_pattern']
ffn_pre_act = cache['blocks.0.ffn.hook_pre_act']
router_probs = cache['ffn.hook_router_probs']  # MoE only
```

### `__init__.py`

Exports: `HookedOneLayerTransformer`, `HookedFFN`, `HookedGLU`, `HookedMHA`, `HookedMoE`, `HookedMoEGLU`
