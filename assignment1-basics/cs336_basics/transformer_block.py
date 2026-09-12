import torch
from torch import nn, Tensor
from jaxtyping import Float, Bool, Int
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.positionwise_feedforward import FFN
from cs336_basics.multihead_self_attention import multihead_self_attention

class transformer_block(nn.Module):
    def __init__(self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()

        # 定义RMSNorm、因果多头自注意力、前馈网络
        self.ln1 = RMSNorm(d_model)
        self.attn = multihead_self_attention(d_model, num_heads, theta, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        
    def forward(self, x: torch.Tensor, token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        y1 = x + self.attn(self.ln1(x), token_positions)
        y2 = y1 + self.ffn(self.ln2(y1))
        return y2
