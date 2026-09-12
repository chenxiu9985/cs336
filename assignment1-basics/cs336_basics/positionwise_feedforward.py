import torch
from torch import nn
from cs336_basics.linear import linear

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None,):
        super().__init__()
        self.w1 = linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.w1(x)
        gate = z * torch.sigmoid(z)  # SiLU
        return self.w2(gate * self.w3(x))
