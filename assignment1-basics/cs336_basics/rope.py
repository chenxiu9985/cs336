import torch
from torch import nn
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        # 不要真的在里维护一个 d×d 的旋转矩阵
        super().__init__()

        # 计算平面旋转角度 θ_k，一共有d_k /2 个平面，k 取值 0,.., d_k/2-1
        # 计算公式 θ_k = θ^(-2k/d_k)
        k = torch.arange(d_k // 2, device=device)
        θ_k = theta ** (-2 * k / d_k)

        # 计算具体位置旋转角度 θ_m，m 取值 0,.., max_seq_len - 1
        m = torch.arange(max_seq_len, device=device)
        θ_m = m[:, None] * θ_k[None, :]

        # 计算这些配置的正余弦
        cos = torch.cos(θ_m)
        sin = torch.sin(θ_m)
        
        # 配置缓存，他们不是可学习参数
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: [..., sequence_length, d_k]
        seq_len = x.shape[-2]

        # 如果没有显式给位置，则默认使用 0,1,...,seq_len-1
        if token_positions is None:
            token_positions = torch.arange(
                seq_len,
                device=x.device
            )
        
        # shape 变化：[..., d_k] 转成拆成多组二维 [..., d_k // 2, 2]
        x = rearrange(x, "... (d two) -> ... d two", two=2)
        x1 = x[..., 0]
        x2 = x[..., 1]

        # 取出缓存
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 如果 x 包含额外的 head 维，需要让 cos/sin 对 head 广播
        while cos.ndim < x1.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # 计算旋转
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        # 合并这些向量
        y = torch.stack([y1, y2], dim=-1)
        return rearrange(y, "... d two -> ... (d two)")