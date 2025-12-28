import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformer_lens.hook_points import HookPoint


class HookedFFN(nn.Module):
    """
    FFN with hook points for analyzing intermediate activations.
    
    Hook points:
        - hook_pre: Input to the FFN
        - hook_pre_act: After up projection, before activation
        - hook_post_act: After activation
        - hook_out: Final output
    """
    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int, activation=nn.SiLU):
        super().__init__()
    
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.activation = activation()

        self.up_proj = nn.Linear(input_dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, output_dim)
        
        # Hook points
        self.hook_pre = HookPoint()
        self.hook_pre_act = HookPoint()
        self.hook_post_act = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_pre(x)
        
        pre_act = self.up_proj(x)
        pre_act = self.hook_pre_act(pre_act)
        
        post_act = self.activation(pre_act)
        post_act = self.hook_post_act(post_act)
        
        out = self.down_proj(post_act)
        out = self.hook_out(out)
    
        return out


class HookedGLU(nn.Module):
    """
    GLU (Gated Linear Unit) with hook points for analyzing intermediate activations.
    
    Hook points:
        - hook_pre: Input to the GLU
        - hook_gate_pre: Gate projection output, before activation
        - hook_gate_post: Gate after activation
        - hook_up: Up projection output
        - hook_fuse: After gating (up * gate)
        - hook_out: Final output
    """
    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int, activation=nn.SiLU):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.activation = activation()

        self.gate_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, output_dim, bias=False)
        
        # Hook points
        self.hook_pre = HookPoint()
        self.hook_gate_pre = HookPoint()
        self.hook_gate_post = HookPoint()
        self.hook_up = HookPoint()
        self.hook_fuse = HookPoint()
        self.hook_out = HookPoint()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_pre(x)
        
        gate = self.gate_proj(x)
        gate = self.hook_gate_pre(gate)
        
        gate = self.activation(gate)
        gate = self.hook_gate_post(gate)

        proj = self.up_proj(x)
        proj = self.hook_up(proj)
        
        fuse = proj * gate
        fuse = self.hook_fuse(fuse)
        
        out = self.down_proj(fuse)
        out = self.hook_out(out)

        return out


class HookedMHA(nn.Module):
    """
    Multi-Head Attention with hook points for analyzing intermediate activations.
    
    Hook points:
        - hook_q: Query after projection and reshaping (B, H, T, D)
        - hook_k: Key after projection and reshaping (B, H, T, D)
        - hook_v: Value after projection and reshaping (B, H, T, D)
        - hook_pattern: Attention pattern/weights (B, H, T, T)
        - hook_z: Attention output before reshape (B, H, T, D)
        - hook_attn_out: Final output after o_proj (B, T, C)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        is_causal: bool = True,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Hook points
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn_out = HookPoint()

    def _shape(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # (B, T, C) -> (B, H, T, D)
        return x.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

    def forward(
        self, 
        x_q: torch.Tensor, 
        x_kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        B, L, C = x_q.shape
        if x_kv is None:
            x_kv = x_q
        _, S, _ = x_kv.shape

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)
        
        q = self._shape(q, B, L)  # (B, H, L, D)
        k = self._shape(k, B, S)  # (B, H, S, D)
        v = self._shape(v, B, S)  # (B, H, S, D)
        
        # Apply hooks
        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        scale = 1.0 / (self.d_head ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, S)
        
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(L, S, dtype=torch.bool, device=q.device), 
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_pattern = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout during training
        if self.training and self.dropout > 0:
            attn_pattern = F.dropout(attn_pattern, p=self.dropout)
        
        attn_pattern = self.hook_pattern(attn_pattern)
        
        # Compute attention output
        z = torch.matmul(attn_pattern, v)  # (B, H, L, D)
        z = self.hook_z(z)
        
        # Merge heads
        y = z.transpose(1, 2).contiguous().view(B, L, C)
        y = self.o_proj(y)
        y = self.hook_attn_out(y)

        return y

