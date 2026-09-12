import torch
from einops import rearrange, einsum

# 使用 einops.einsum 进行批量矩阵乘法
# 常规使用torch，虽然简单但是对于输入、输出和形式都比较模糊
D = torch.randn(2, 3, 4)
A = torch.randn(6, 4)
Y1 = D @ A.T
print("Y1形状：" + str(Y1.shape))

# 使用einsum，表明输入的形状和输出的形状
Y2 = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
Y3 = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
print("Y2形状：" + str(Y2.shape))
print("Y3形状：" + str(Y3.shape))

# 使用 einops.rearrange 进行广播式操作
images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10) # 0, 1/9, 2/9, ... ,1 十个亮度系数
print(dim_by)

dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
print(dim_value)
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

dimmed_images = einsum(images, dim_by, "batch height width channel, dim_value -> batch dim_value height width channel")

# 使用 einops.rearrange 实现像素混合