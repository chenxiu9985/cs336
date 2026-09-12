import torch
from torch import nn, Tensor
from jaxtyping import Float, Bool, Int
from cs336_basics.linear import linear
from cs336_basics.softmax import softmax
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.embedding import embedding
from cs336_basics.transformer_block import transformer_block

class transformer_lm(nn.Module):
    def __init__(self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        # 定义embedding模块
        self.token_embeddings = embedding(vocab_size, d_model)
        # 定义transformer块组
        self.layers = nn.ModuleList([
            transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
            )
            for _ in range(num_layers)
        ])
        # 定义RMSNorm
        self.ln_final = RMSNorm(d_model)
        # 定义Linear映射
        self.lm_head = linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
        
