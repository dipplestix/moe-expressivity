import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from components import MHA, FFN, GLU, MoE, MoEGLU, resolve_activation


class OneLayerTransformer(nn.Module):
    def __init__(self,
    model_dim,
    num_heads,
    ffn_type="ffn",
    dropout=0.0,
    vocab_size=10,
    max_seq_len=128,
    use_norm=True,
    is_causal=True,
    tie_embeddings=True,
    activation="silu",
    intermediate_dim=None,
    num_experts=4,
    top_k=1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ffn_type = ffn_type
        self.dropout = dropout
        self.use_norm = use_norm
        self.is_causal = is_causal
        self.tie_embeddings = tie_embeddings

        act_cls = resolve_activation(activation)

        self.vocab = nn.Embedding(vocab_size, model_dim)
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)

        if not tie_embeddings:
            self.unembed = nn.Linear(model_dim, vocab_size, bias=False)

        self.atn_norm = nn.RMSNorm(model_dim) if use_norm else nn.Identity()
        self.atn = MHA(d_model=model_dim, num_heads=num_heads, dropout=dropout, is_causal=is_causal)

        self.ffn_norm = nn.RMSNorm(model_dim) if use_norm else nn.Identity()
        if ffn_type == "ffn":
            idim = intermediate_dim if intermediate_dim is not None else model_dim * 4
            self.ffn = FFN(input_dim=model_dim, intermediate_dim=idim, output_dim=model_dim, activation=act_cls)
        elif ffn_type == "glu":
            idim = intermediate_dim if intermediate_dim is not None else model_dim * 2
            self.ffn = GLU(input_dim=model_dim, intermediate_dim=idim, output_dim=model_dim, activation=act_cls)
        elif ffn_type == "moe":
            idim = intermediate_dim if intermediate_dim is not None else model_dim * 4
            self.ffn = MoE(input_dim=model_dim, intermediate_dim=idim, output_dim=model_dim,
                           num_experts=num_experts, top_k=top_k, activation=act_cls)
        elif ffn_type == "moe_glu":
            idim = intermediate_dim if intermediate_dim is not None else model_dim * 4
            self.ffn = MoEGLU(input_dim=model_dim, intermediate_dim=idim, output_dim=model_dim,
                           num_experts=num_experts, top_k=top_k, activation=act_cls)
        else:
            raise ValueError(f"Invalid FFN type: {ffn_type}")

        self.out_norm = nn.RMSNorm(model_dim) if use_norm else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.vocab(x) + self.pos_embed(pos)

        x = x + self.resid_dropout(self.atn(self.atn_norm(x)))

        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))

        if self.ffn_type in ("moe", "moe_glu"):
            self._aux_loss = self.ffn._aux_loss

        x = self.out_norm(x)

        if self.tie_embeddings:
            logits = F.linear(x, self.vocab.weight)
        else:
            logits = self.unembed(x)
        return logits
