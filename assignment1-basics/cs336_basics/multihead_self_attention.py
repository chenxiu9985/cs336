import torch
from torch import nn, Tensor
from einops import rearrange
from jaxtyping import Float, Bool, Int
from cs336_basics.linear import linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class multihead_self_attention(nn.Module):
    def __init__(self,
        d_model: int, 
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        # 查询、键、值
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # # Q/K/V 一次生成所有 heads
        self.q_proj = linear(d_model, d_model)
        self.k_proj = linear(d_model, d_model)
        self.v_proj = linear(d_model, d_model)
        self.output_proj = linear(d_model, d_model)

        # 旋转位置编码
        self.rope = None
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        # 计算自注意力，并拆分成多头head
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # 使用旋转位置编码
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 构造causal mask
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # 计算缩放点积注意力
        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = rearrange(attention, "... h seq d -> ... seq (h d)")
        
        return self.output_proj(attention)