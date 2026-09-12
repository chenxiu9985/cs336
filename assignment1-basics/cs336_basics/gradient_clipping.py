import torch
from typing import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6,):
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return

    # 求L2范数
    total_squared_norm = torch.stack([grad.square().sum for grad in grads]).sum()
    l2_norm = torch.sqrt(total_squared_norm)

    if l2_norm >= max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        with torch.no_grad():
            for grad in grads:
                grad.mul_(scale)