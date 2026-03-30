import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def resolve_activation(activation):
    """Resolve activation from string or nn.Module class."""
    if isinstance(activation, str):
        return {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[activation.lower()]
    return activation


class FFN(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, activation=nn.SiLU):
        super().__init__()
    
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.activation = activation()

        self.up_proj = nn.Linear(input_dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, output_dim) 

    def forward(self, x):
        proj = self.activation(self.up_proj(x))
        out = self.down_proj(proj)
    
        return out


class GLU(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, activation=nn.SiLU):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.activation = activation()

        self.gate_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, output_dim, bias=False)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        gate = self.activation(gate)

        proj = self.up_proj(x)
        
        fuse = proj*gate
        out = self.down_proj(fuse)

        return out


class MHA(nn.Module):
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

    def _shape(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # (B, T, C) -> (B, H, T, D)
        return x.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

    def forward(
        self, 
        x_q: torch.Tensor, 
        x_kv: Optional[torch.Tensor] = None
    ):
        
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

        y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal
        )  # (B, H, L, D)

        
        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.o_proj(y)

        return y


class MoE(nn.Module):
    """Mixture of Experts layer with Switch-style top-k routing."""

    def __init__(self, input_dim, intermediate_dim, output_dim, num_experts=4, top_k=1, activation=nn.SiLU):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(input_dim, num_experts, bias=False)

        expert_intermediate = intermediate_dim // num_experts
        self.experts = nn.ModuleList([
            FFN(input_dim, expert_intermediate, output_dim, activation=activation)
            for _ in range(num_experts)
        ])

        self._aux_loss = torch.tensor(0.0)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        router_logits = self.router(x_flat)  # (B*T, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (B*T, E)

        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (B*T, k)
        #check need to normalize?
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renormalize

        # Load-balancing loss: E * sum(f_i * P_i)
        # f_i = fraction of tokens routed to expert i
        one_hot = F.one_hot(topk_indices, self.num_experts).float()  # (B*T, k, E)
        tokens_per_expert = one_hot.sum(dim=1).mean(dim=0)  # (E,) fraction routed
        prob_per_expert = router_probs.mean(dim=0)  # (E,) mean probability
        self._aux_loss = self.num_experts * (tokens_per_expert * prob_per_expert).sum()

        # Dispatch to experts
        out = torch.zeros(B * T, self.experts[0].output_dim, device=x.device, dtype=x.dtype)
        for k_idx in range(self.top_k):
            for e_idx in range(self.num_experts):
                mask = topk_indices[:, k_idx] == e_idx
                if mask.any():
                    expert_out = self.experts[e_idx](x_flat[mask])
                    out[mask] += topk_weights[mask, k_idx].unsqueeze(-1) * expert_out

        return out.view(B, T, -1)

class MoEGLU(nn.Module):
    "MoE-glu"
    def __init__(self, input_dim, intermediate_dim, output_dim, num_experts=4, top_k=1, activation=nn.SiLU):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(input_dim, num_experts, bias=False)

        expert_intermediate = intermediate_dim // num_experts
        self.experts = nn.ModuleList([
            GLU(input_dim, expert_intermediate, output_dim, activation=activation)
            for _ in range(num_experts)
        ])

        self._aux_loss = torch.tensor(0.0)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        router_logits = self.router(x_flat)  # (B*T, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (B*T, E)

        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (B*T, k)
        #check need to normalize?
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renormalize

        # Load-balancing loss: E * sum(f_i * P_i)
        # f_i = fraction of tokens routed to expert i
        one_hot = F.one_hot(topk_indices, self.num_experts).float()  # (B*T, k, E)
        tokens_per_expert = one_hot.sum(dim=1).mean(dim=0)  # (E,) fraction routed
        prob_per_expert = router_probs.mean(dim=0)  # (E,) mean probability
        self._aux_loss = self.num_experts * (tokens_per_expert * prob_per_expert).sum()

        # Dispatch to experts
        out = torch.zeros(B * T, self.experts[0].output_dim, device=x.device, dtype=x.dtype)
        for k_idx in range(self.top_k):
            for e_idx in range(self.num_experts):
                mask = topk_indices[:, k_idx] == e_idx
                if mask.any():
                    expert_out = self.experts[e_idx](x_flat[mask])
                    out[mask] += topk_weights[mask, k_idx].unsqueeze(-1) * expert_out

        return out.view(B, T, -1)





    

