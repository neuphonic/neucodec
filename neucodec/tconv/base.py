import einops
import torch
from torch import nn
import torch.nn.functional as F

from ..distill_layers import Conv1d


def trend_pool(x, kernel_size):
    if kernel_size > 1:
        pool_args = dict(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return F.avg_pool1d(F.max_pool1d(x.abs(), **pool_args), **pool_args)
        # return F.avg_pool1d(F.max_pool1d(x, **pool_args), **pool_args)  # woabs
    else:
        return x


class TrendPool(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return trend_pool(x, self.kernel_size)


class FirstBlock(nn.Module):
    def __init__(
        self,
        target_dim,
        conv_kernels=(7, 7, 7, 7),
        pool_kernels=(1, 3, 5, 9),
        dilation_rate=2,
    ):
        super().__init__()
        assert target_dim % len(pool_kernels) == 0
        each_dim = target_dim // len(pool_kernels)
        blocks = []
        for conv_kernel, pool_kernel in zip(conv_kernels, pool_kernels):
            conv_dilation = pool_kernel // dilation_rate + 1
            conv_padding = (conv_kernel - 1) * conv_dilation // 2
            blocks.append(
                nn.Sequential(
                    TrendPool(pool_kernel),
                    Conv1d(
                        1,
                        each_dim,
                        kernel_size=conv_kernel,
                        dilation=conv_dilation,
                        padding=conv_padding,
                    ),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        return torch.cat([block(x) for block in self.blocks], dim=1)


class EnhanceBlock(FirstBlock):
    def __init__(self, dim):
        super().__init__(4, conv_kernels=(7, 7, 7, 7), pool_kernels=(1, 3, 5, 9))
        self.dim = dim
        self.merge_layer = nn.Sequential(
            # nn.LeakyReLU(),  # ! if active or use InstanceNorm1d
            nn.InstanceNorm1d(4, affine=True),
            nn.Conv1d(4, 1, kernel_size=1),
        )

    def forward(self, x):
        x = einops.rearrange(x, "b c t -> (b c) 1 t", c=self.dim)
        y = super().forward(x)
        y = self.merge_layer(y)
        y = einops.rearrange(y, "(b c) 1 t -> b c t", c=self.dim)
        return y  # ! x + y or x + y * x


class SimpleEnhanceBlock(FirstBlock):
    def __init__(self, dim):
        super().__init__(4, conv_kernels=(7, 7, 7, 7), pool_kernels=(1, 3, 5, 9))
        self.dim = dim
        self.merge_layer = nn.Sequential(
            # nn.LeakyReLU(),  # ! if active or use InstanceNorm1d
            nn.InstanceNorm1d(4, affine=True),
            nn.Conv1d(4, self.dim, kernel_size=1),
        )

    def forward(self, x):
        xi = x[:, :1, :]
        yi = super().forward(xi)
        y = self.merge_layer(yi)
        return x + y * x
