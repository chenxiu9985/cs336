import torch
import math
from typing import Optional
from collections.abc import Callable, Iterable

class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0.01,):
        beta1, beta2 = betas

        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable]=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]

            # 对于每一个参数
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 如何维护状态v，m，t
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]
            
                # 计算梯度
                grad = p.grad.data
                # 计算自适应学习率 alpha_t
                alpha_t = lr * math.sqrt(1-beta2**t) / (1-beta1**t)
                # 权重衰减
                p.mul_(1 - lr * weight_decay)
                # 更新一、二阶矩估计，原地更新
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 更新参数
                p.addcdiv_(m, v.sqrt() + epsilon, value=-alpha_t,)
        return loss