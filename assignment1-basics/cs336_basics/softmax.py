import torch 

def softmax(x: torch.Tensor, dim: int):
    # 沿指定维度找到每组最大值，并保留该维度以便后续广播
    max_value = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_value
    exp_x = torch.exp(x)
    # 沿同一个维度求和，作为 softmax 分母
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x