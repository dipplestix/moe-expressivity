import torch
import torch.nn as nn
import torch.nn.functional as F
from components import MHA, FFN, GLU


class OneLayerTransformer(nn.Module):
    def __init__(self, 
    model_dim, 
    num_heads, 
    ffn_type="ffn", 
    dropout=0.0, 
    vocab_size=10,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ffn_type = ffn_type
        self.dropout = dropout

        self.vocab = nn.Embedding(vocab_size, model_dim)

        self.atn_norm = nn.RMSNorm(model_dim)
        self.atn = MHA(d_model=model_dim, num_heads=num_heads, dropout=dropout)

        self.ffn_norm = nn.RMSNorm(model_dim)
        if ffn_type == "ffn":   
            intermediate_dim = model_dim * 4
            self.ffn = FFN(input_dim=model_dim, intermediate_dim=intermediate_dim, output_dim=model_dim, activation=nn.SiLU())
        elif ffn_type == "glu":
            intermediate_dim = model_dim * 2
            self.ffn = GLU(input_dim=model_dim, intermediate_dim=intermediate_dim, output_dim=model_dim, activation=nn.SiLU())
        else:
            raise ValueError(f"Invalid FFN type: {ffn_type}")

        self.out_norm = nn.RMSNorm(model_dim)


    def forward(self, x):
        x = self.vocab(x)
        
        x = self.atn_norm(x)
        x = x + self.atn(x)

        x = self.ffn_norm(x)
        x = x + self.ffn(x)
        
        x = self.out_norm(x)
        
        logits = F.linear(x, self.vocab.weight)
        return logits