import torch.nn as nn

from torch.nn.utils import weight_norm

from .activations import SnakeBeta
from .alias_free_torch import Activation1d


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Activation1d(activation=SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Activation1d(activation=SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, dilations=(1, 3, 9)):
        super().__init__()
        runits = [ResidualUnit(dim // 2, dilation=d) for d in dilations]
        self.block = nn.Sequential(
            *runits,
            Activation1d(activation=SnakeBeta(dim // 2, alpha_logscale=True)),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
            ),
        )

    def forward(self, x):
        return self.block(x)


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        code_dim: int,
        encode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super(SemanticEncoder, self).__init__()

        self.initial_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.residual_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
        )

        self.final_conv = nn.Conv1d(
            in_channels=encode_channels,
            out_channels=code_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x):
        x = self.initial_conv(x)  # (Batch, Encode_channels, Length)
        x = self.residual_blocks(x) + x  # 残差连接
        x = self.final_conv(x)  # (Batch, Code_dim, Length)
        return x
