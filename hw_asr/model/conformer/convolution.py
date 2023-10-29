import torch
from torch import nn


class ConvolutionalModule(nn.Module):
    """
    https://towardsdatascience.com/efficient-image-segmentation-using-pytorch-part-3-3534cf04fb89#:~:text=A%20depthwise%20grouped%20convolution%2C%20where,called%20a%20%E2%80%9Cgrouped%E2%80%9D%20convolution.
    """

    def __init__(self, embed_dim, dropout, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1, "for mathcing in and out dim of depthwise conv"
        self.norm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size,
            groups=embed_dim,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(embed_dim)
        self.silu = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        # print("ConvolutionalModule")
        # print(f"{x.shape=}")
        output = self.norm(x)
        output = output.transpose(-1, -2)
        # print(f"{output.shape=}")
        output = self.pointwise_conv1(output)
        # print(f"after pointwise {output.shape=}")
        output = self.glu(output)
        # print(f"{output.shape=}")
        output = self.depthwise_conv(output)
        # print(f"{output.shape=}")
        output = self.bn(output)
        # print(f"{output.shape=}")
        output = self.silu(output)
        # print(f"{output.shape=}")
        output = self.pointwise_conv2(output)
        # print(f"{output.shape=}")
        output = self.dropout(output).transpose(-1, -2) + x
        # print(f"{output.shape=}")
        return output
