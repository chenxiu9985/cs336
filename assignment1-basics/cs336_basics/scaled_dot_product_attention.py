import torch
from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    # 计算注意力分数，使用缩放点积注意力
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scaled_dot = QK / torch.sqrt(torch.tensor(d_k, device=Q.device))

    # mask操作
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(~mask, float("-inf"))

    # 进行softmax操作，得到注意力权重
    weight = softmax(scaled_dot, dim=-1)

    # 返回注意力
    return einsum(weight, V, "... queries keys, ... keys d_v -> ... queries d_v")