import torch
from torch import nn
from einops import einsum

class linear(nn.Module):
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None=None, 
        dtype: torch.dtype | None=None
    ):
        # 调用父类构造器
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        # 初始化
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
    