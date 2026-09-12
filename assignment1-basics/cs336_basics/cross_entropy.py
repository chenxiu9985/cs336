import torch
from torch import Tensor
from jaxtyping import Float, Int
# from cs336_basics.softmax import softmax

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], 
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    # 第一版，强行计算softmax，会造成下溢，softmax里面做了上溢处理
    # batch_idx = torch.arange(inputs.shape[0])
    # p = softmax(inputs, -1)[batch_idx, targets]
    # loss = torch.mean(-torch.log(p))

    # 第二版，整体公式优化，基本保留softmax的操作
    max_values = torch.max(inputs, dim=-1, keepdim=True).values
    inputs = inputs - max_values
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs), dim=-1))

    batch_idx = torch.arange(inputs.shape[0], device=inputs.device)
    loss = torch.mean(log_sum_exp - inputs[batch_idx, targets])

    return loss