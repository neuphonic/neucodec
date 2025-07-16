import torch
import numpy as np

from torch import nn

from .module import WNConv1d, EncoderBlock
from .alias_free_torch import Activation1d
from . import activations


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class CodecEncoder(nn.Module):
    def __init__(
        self,
        ngf=48,
        up_ratios=[2, 2, 4, 4, 5],
        dilations=(1, 3, 9),
        hidden_dim=1024,
        depth=12,
        heads=12,
        pos_meb_dim=64,
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        d_model = ngf
        self.conv_blocks = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.conv_blocks += [
                EncoderBlock(d_model, stride=stride, dilations=dilations)
            ]

        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        self.conv_final_block = [
            Activation1d(
                activation=activations.SnakeBeta(d_model, alpha_logscale=True)
            ),
            WNConv1d(d_model, hidden_dim, kernel_size=3, padding=1),
        ]
        self.conv_final_block = nn.Sequential(*self.conv_final_block)

        self.reset_parameters()

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.conv_final_block(x)
        x = x.permute(0, 2, 1)
        return x

    def inference(self, x):
        return self.block(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)
