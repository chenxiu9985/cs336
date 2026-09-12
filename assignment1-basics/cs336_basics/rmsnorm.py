import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入数据提升为 `torch.float32` 类型，以防止在对输入进行平方运算时发生溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 计算 x 沿最后一个维度计算平方的均值
        mean_square = x.square().mean(dim=-1, keepdim=True)
        # 完整 RMSNorm(x)
        result = x * torch.rsqrt(mean_square + self.eps) * self.weight
        return result.to(in_dtype)