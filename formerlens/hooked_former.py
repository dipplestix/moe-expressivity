import math
import sys
from typing import Dict, List, Optional, Callable, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformer_lens.hook_points import HookPoint, HookedRootModule

sys.path.insert(0, "model")
from components import resolve_activation

from .hooked_components import HookedMHA, HookedFFN, HookedGLU, HookedMoE, HookedMoEGLU


class HookedOneLayerTransformer(HookedRootModule):
    """
    A hooked version of the OneLayerTransformer that exposes all intermediate
    activations via TransformerLens hook points for interpretability analysis.

    Hook points (in order of forward pass):
        - hook_embed: After token + positional embedding
        - hook_attn_pre: Before attention (after attn norm)
        - blocks.0.attn.hook_q/k/v/pattern/z: Attention internals
        - blocks.0.attn.hook_attn_out: Attention output
        - hook_resid_mid: Residual stream after attention
        - hook_ffn_pre: Before FFN (after ffn norm)
        - blocks.0.ffn.hook_pre/pre_act/post_act/out: FFN internals
        - hook_resid_post: Residual stream after FFN
        - hook_logits: Final logits

    For MoE (ffn_type="moe"), additional hooks:
        - blocks.0.ffn.hook_router_logits/router_probs
        - blocks.0.ffn.hook_expert_selection/expert_weights
        - blocks.0.ffn.experts.{i}.hook_pre_act/post_act
        - blocks.0.ffn.hook_out
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_type: str = "ffn",
        dropout: float = 0.0,
        vocab_size: int = 10,
        max_seq_len: int = 128,
        use_norm: bool = True,
        is_causal: bool = True,
        tie_embeddings: bool = True,
        activation: str = "silu",
        intermediate_dim: Optional[int] = None,
        num_experts: int = 4,
        top_k: int = 1,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ffn_type = ffn_type
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_norm = use_norm
        self.is_causal = is_causal
        self.tie_embeddings = tie_embeddings

        act_cls = resolve_activation(activation)

        # Embeddings
        self.vocab = nn.Embedding(vocab_size, model_dim)
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)

        if not tie_embeddings:
            self.unembed = nn.Linear(model_dim, vocab_size, bias=False)

        # Attention block
        self.atn_norm = nn.RMSNorm(model_dim) if use_norm else nn.Identity()
        self.atn = HookedMHA(d_model=model_dim, num_heads=num_heads, dropout=dropout, is_causal=is_causal)

        # FFN block
        self.ffn_norm = nn.RMSNorm(model_dim) if use_norm else nn.Identity()
        if ffn_type == "ffn":
            idim = intermediate_dim if intermediate_dim is not None else model_dim * 4
            self.ffn = HookedFFN(input_dim=model_dim, intermediate_dim=idim, output_dim=model_dim, activation=act_cls)
        elif ffn_type == "glu":
            idim = intermediate_dim if intermediate_dim is not None else model_dim * 2
            self.ffn = HookedGLU(input_dim=model_dim, intermediate_dim=idim, output_dim=model_dim, activation=act_cls)
        elif ffn_type == "moe":
            idim = intermediate_dim if intermediate_dim is not None else model_dim * 4
            self.ffn = HookedMoE(
                input_dim=model_dim, intermediate_dim=idim, output_dim=model_dim,
                num_experts=num_experts, top_k=top_k, activation=act_cls,
            )
        elif ffn_type == "moe_glu":
            idim = intermediate_dim if intermediate_dim is not None else model_dim * 4
            self.ffn = HookedMoEGLU(
                input_dim=model_dim, intermediate_dim=idim, output_dim=model_dim,
                num_experts=num_experts, top_k=top_k, activation=act_cls,
            )
        else:
            raise ValueError(f"Invalid FFN type: {ffn_type}")

        # Output norm
        self.out_norm = nn.RMSNorm(model_dim) if use_norm else nn.Identity()

        self._aux_loss = torch.tensor(0.0)

        # Hook points for residual stream
        self.hook_embed = HookPoint()
        self.hook_attn_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_ffn_pre = HookPoint()
        self.hook_resid_post = HookPoint()
        self.hook_logits = HookPoint()

        # Setup hook points dict for HookedRootModule
        self.setup()

    def forward(
        self,
        x: torch.Tensor,
        return_type: Optional[str] = "logits"
    ) -> Union[torch.Tensor, None]:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)

        # Embeddings
        resid = self.vocab(x) + self.pos_embed(pos)
        resid = self.hook_embed(resid)

        # Attention block
        attn_in = self.atn_norm(resid)
        attn_in = self.hook_attn_pre(attn_in)
        attn_out = self.atn(attn_in)
        resid = resid + attn_out
        resid = self.hook_resid_mid(resid)

        # FFN block
        ffn_in = self.ffn_norm(resid)
        ffn_in = self.hook_ffn_pre(ffn_in)
        ffn_out = self.ffn(ffn_in)
        resid = resid + ffn_out
        resid = self.hook_resid_post(resid)

        if self.ffn_type in ("moe", "moe_glu"):
            self._aux_loss = self.ffn._aux_loss

        # Output
        resid = self.out_norm(resid)
        if self.tie_embeddings:
            logits = F.linear(resid, self.vocab.weight)
        else:
            logits = self.unembed(resid)
        logits = self.hook_logits(logits)

        if return_type == "logits":
            return logits
        elif return_type is None:
            return None
        else:
            raise ValueError(f"Invalid return_type: {return_type}")

    @classmethod
    def from_pretrained(
        cls,
        original_model: nn.Module,
        strict: bool = True
    ) -> "HookedOneLayerTransformer":
        """
        Create a HookedOneLayerTransformer from an existing OneLayerTransformer,
        copying all weights.
        """
        hooked_model = cls(
            model_dim=original_model.model_dim,
            num_heads=original_model.num_heads,
            ffn_type=original_model.ffn_type,
            dropout=original_model.dropout,
            vocab_size=original_model.vocab.num_embeddings,
            max_seq_len=original_model.pos_embed.num_embeddings,
            use_norm=original_model.use_norm,
            is_causal=getattr(original_model, "is_causal", True),
            tie_embeddings=getattr(original_model, "tie_embeddings", True),
        )

        # Copy all matching weights
        src_sd = original_model.state_dict()
        tgt_sd = hooked_model.state_dict()
        for key in src_sd:
            if key in tgt_sd:
                tgt_sd[key] = src_sd[key]
        hooked_model.load_state_dict(tgt_sd, strict=False)

        return hooked_model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        vocab_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> "HookedOneLayerTransformer":
        """
        Load a HookedOneLayerTransformer from a checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        config = checkpoint.get("config", {})
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        if vocab_size is None:
            if "vocab.weight" in state_dict:
                vocab_size = state_dict["vocab.weight"].shape[0]
            else:
                vocab_size = config.get("vocab_size", 12)

        if max_seq_len is None:
            if "pos_embed.weight" in state_dict:
                max_seq_len = state_dict["pos_embed.weight"].shape[0]
            else:
                max_seq_len = config.get("max_seq_len", 128)

        hooked_model = cls(
            model_dim=config.get("model_dim", 64),
            num_heads=config.get("num_heads", 4),
            ffn_type=config.get("ffn_type", "ffn"),
            dropout=config.get("dropout", 0.0),
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            use_norm=config.get("use_norm", True),
            is_causal=config.get("is_causal", True),
            tie_embeddings=config.get("tie_embeddings", True),
            activation=config.get("activation", "silu"),
            intermediate_dim=config.get("intermediate_dim", None),
            num_experts=config.get("num_experts", 4),
            top_k=config.get("top_k", 1),
        )

        # Map original model keys to hooked model keys
        hooked_state_dict = {}
        for key, value in state_dict.items():
            if "hook_" in key:
                continue
            hooked_state_dict[key] = value

        hooked_model.load_state_dict(hooked_state_dict, strict=False)
        hooked_model.to(device)

        return hooked_model

    def get_hook_names(self) -> List[str]:
        """Return a list of all hook point names in the model."""
        return list(self.hook_dict.keys())

    def print_hooks(self):
        """Print all available hook points."""
        print("Available hook points:")
        for name in sorted(self.get_hook_names()):
            print(f"  - {name}")
