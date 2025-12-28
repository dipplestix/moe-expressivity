import math
from typing import Dict, List, Optional, Callable, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformer_lens.hook_points import HookPoint, HookedRootModule

from .hooked_components import HookedMHA, HookedFFN, HookedGLU


class HookedOneLayerTransformer(HookedRootModule):
    """
    A hooked version of the OneLayerTransformer that exposes all intermediate
    activations via TransformerLens hook points for interpretability analysis.
    
    Hook points (in order of forward pass):
        - hook_embed: After token + positional embedding
        - hook_attn_pre: Before attention (after attn norm)
        - blocks.0.attn.hook_q: Query vectors (B, H, T, D)
        - blocks.0.attn.hook_k: Key vectors (B, H, T, D)
        - blocks.0.attn.hook_v: Value vectors (B, H, T, D)
        - blocks.0.attn.hook_pattern: Attention pattern (B, H, T, T)
        - blocks.0.attn.hook_z: Pre-output attention (B, H, T, D)
        - blocks.0.attn.hook_attn_out: Attention output
        - hook_resid_mid: Residual stream after attention
        - hook_ffn_pre: Before FFN (after ffn norm)
        - blocks.0.ffn.hook_pre: FFN input
        - blocks.0.ffn.hook_pre_act / hook_gate_pre: Before activation
        - blocks.0.ffn.hook_post_act / hook_gate_post: After activation
        - blocks.0.ffn.hook_out: FFN output
        - hook_resid_post: Residual stream after FFN
        - hook_logits: Final logits
        
    Usage:
        model = HookedOneLayerTransformer(model_dim=64, num_heads=4)
        
        # Run with hooks
        logits, cache = model.run_with_cache(tokens)
        
        # Access cached activations
        attention_pattern = cache["blocks.0.attn.hook_pattern"]
        ffn_activations = cache["blocks.0.ffn.hook_post_act"]
        
        # Run with custom hook function
        def my_hook(activation, hook):
            print(f"{hook.name}: {activation.shape}")
            return activation
        
        model.run_with_hooks(tokens, fwd_hooks=[("hook_embed", my_hook)])
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
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ffn_type = ffn_type
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_norm = use_norm
        
        # Embeddings
        self.vocab = nn.Embedding(vocab_size, model_dim)
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)
        
        # Attention block
        self.atn_norm = nn.RMSNorm(model_dim) if use_norm else nn.Identity()
        self.atn = HookedMHA(d_model=model_dim, num_heads=num_heads, dropout=dropout)
        
        # FFN block
        self.ffn_norm = nn.RMSNorm(model_dim) if use_norm else nn.Identity()
        if ffn_type == "ffn":
            intermediate_dim = model_dim * 4
            self.ffn = HookedFFN(
                input_dim=model_dim, 
                intermediate_dim=intermediate_dim, 
                output_dim=model_dim, 
                activation=nn.SiLU
            )
        elif ffn_type == "glu":
            intermediate_dim = model_dim * 2
            self.ffn = HookedGLU(
                input_dim=model_dim, 
                intermediate_dim=intermediate_dim, 
                output_dim=model_dim, 
                activation=nn.SiLU
            )
        else:
            raise ValueError(f"Invalid FFN type: {ffn_type}")
        
        # Output norm
        self.out_norm = nn.RMSNorm(model_dim) if use_norm else nn.Identity()
        
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
        """
        Forward pass through the model.
        
        Args:
            x: Input token indices (B, T)
            return_type: What to return - "logits", "loss", or None
            
        Returns:
            Logits tensor (B, T, vocab_size) or None if return_type is None
        """
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
        
        # Output
        resid = self.out_norm(resid)
        logits = F.linear(resid, self.vocab.weight)
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
        
        Args:
            original_model: The original OneLayerTransformer model
            strict: Whether to require exact weight matching
            
        Returns:
            HookedOneLayerTransformer with copied weights
        """
        hooked_model = cls(
            model_dim=original_model.model_dim,
            num_heads=original_model.num_heads,
            ffn_type=original_model.ffn_type,
            dropout=original_model.dropout,
            vocab_size=original_model.vocab.num_embeddings,
            max_seq_len=original_model.pos_embed.num_embeddings,
            use_norm=original_model.use_norm,
        )
        
        # Copy embeddings
        hooked_model.vocab.load_state_dict(original_model.vocab.state_dict())
        hooked_model.pos_embed.load_state_dict(original_model.pos_embed.state_dict())
        
        # Copy norms
        if original_model.use_norm:
            hooked_model.atn_norm.load_state_dict(original_model.atn_norm.state_dict())
            hooked_model.ffn_norm.load_state_dict(original_model.ffn_norm.state_dict())
            hooked_model.out_norm.load_state_dict(original_model.out_norm.state_dict())
        
        # Copy attention weights
        hooked_model.atn.q_proj.load_state_dict(original_model.atn.q_proj.state_dict())
        hooked_model.atn.k_proj.load_state_dict(original_model.atn.k_proj.state_dict())
        hooked_model.atn.v_proj.load_state_dict(original_model.atn.v_proj.state_dict())
        hooked_model.atn.o_proj.load_state_dict(original_model.atn.o_proj.state_dict())
        
        # Copy FFN weights
        if original_model.ffn_type == "ffn":
            hooked_model.ffn.up_proj.load_state_dict(original_model.ffn.up_proj.state_dict())
            hooked_model.ffn.down_proj.load_state_dict(original_model.ffn.down_proj.state_dict())
        elif original_model.ffn_type == "glu":
            hooked_model.ffn.gate_proj.load_state_dict(original_model.ffn.gate_proj.state_dict())
            hooked_model.ffn.up_proj.load_state_dict(original_model.ffn.up_proj.state_dict())
            hooked_model.ffn.down_proj.load_state_dict(original_model.ffn.down_proj.state_dict())
        
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
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
            vocab_size: Override vocab size (useful if not in checkpoint config)
            max_seq_len: Override max sequence length (useful if not in checkpoint config)
            
        Returns:
            HookedOneLayerTransformer with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Get model config from checkpoint
        config = checkpoint.get("config", {})
        
        # Try to infer vocab_size from embedding weights if not provided
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if vocab_size is None:
            if "vocab.weight" in state_dict:
                vocab_size = state_dict["vocab.weight"].shape[0]
            else:
                vocab_size = config.get("vocab_size", 12)  # Default for add-7 task
        
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
        )
        
        # Map original model keys to hooked model keys
        hooked_state_dict = {}
        for key, value in state_dict.items():
            # Skip hook points from original if any
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
