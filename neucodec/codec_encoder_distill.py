import torch
from torch import Tensor
from torch import nn
from local_attention.transformer import DynamicPositionBias, LocalMHA, FeedForward
from .distill_layers import ChannelNorm, Conv1d, Linear, GRN, Snake1d
from .tconv.t_first import FirstBlock


class LocalTrans(nn.Module):
    def __init__(
        self,
        dim=512,
        depth=6,
        causal=True,
        local_attn_window_size=512,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_dynamic_pos_bias=False,
        qk_rmsnorm=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        self.window_size = local_attn_window_size
        self.use_rotary_pos_emb = not use_dynamic_pos_bias
        self.dynamic_pos_bias = (
            None
            if self.use_rotary_pos_emb
            else DynamicPositionBias(dim=dim // 2, heads=heads)
        )

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LocalMHA(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            causal=causal,
                            window_size=self.window_size,
                            use_xpos=False,
                            xpos_scale_base=None,
                            use_rotary_pos_emb=self.use_rotary_pos_emb,
                            prenorm=True,
                            qk_rmsnorm=qk_rmsnorm,
                            exact_windowsize=False,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x, mask=None):
        attn_bias = (
            None
            if self.use_rotary_pos_emb
            else self.dynamic_pos_bias(self.window_size, self.window_size * 2)
        )
        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_bias=attn_bias) + x
            x = ff(x) + x

        return x

    @classmethod
    def builder(
        cls, feature_dim=128, depth=2, local_window_size=200, use_dynamic_pos_bias=False
    ):
        return cls(
            dim=feature_dim,
            depth=depth,
            dim_head=feature_dim // 4,
            heads=6,
            ff_mult=4,
            causal=True,
            local_attn_window_size=local_window_size,
            use_dynamic_pos_bias=use_dynamic_pos_bias,
        )


class LocalEncoder(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        depth=2,
        local_window_size=200,
        use_dynamic_pos_bias=False,
    ):
        super().__init__()
        self.local_trans = LocalTrans.builder(
            feature_dim=feature_dim,
            depth=depth,
            local_window_size=local_window_size,
            use_dynamic_pos_bias=use_dynamic_pos_bias,
        )

    def forward(self, feature):
        """
        Args:
            feature: (B, C, T)
        Returns:
            local_feature: (B, T, C)
        """
        feature = feature.permute(0, 2, 1)
        feature = self.local_trans(feature)
        return feature


class DownTrans(nn.Module):
    def __init__(
        self, feature_dim=128, window_size=200, compress_rate=2, depth=2, **kwargs
    ):
        super().__init__()
        assert window_size % compress_rate == 0
        self.feature_dim = feature_dim
        self.compress_rate = compress_rate
        self.trans = LocalTrans.builder(
            feature_dim, local_window_size=window_size, depth=depth, **kwargs
        )
        self.down_layer = Conv1d(
            feature_dim, feature_dim, kernel_size=compress_rate, stride=compress_rate
        )

    def forward(self, x):
        x = self.trans(x)
        # x = x[:, ::self.compress_rate, :]  # v1
        x = self.down_layer(x.permute(0, 2, 1)).permute(0, 2, 1)  # v2
        return x


class CompressedLocalEncoderWithCache(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        local_window_size=200,
        compress_rate=2,
        cache_size=3,
        depth=4,
        **kwargs,
    ):
        super().__init__()
        self.local_window_size = local_window_size
        self.cache_size = cache_size
        self.compress_rate = compress_rate
        self.trans_window_size = local_window_size + cache_size

        self.cache_token = nn.Parameter(
            torch.randn(1, self.cache_size * self.compress_rate, feature_dim)
        )

        self.down_trans = DownTrans(
            feature_dim,
            window_size=self.trans_window_size * compress_rate,
            compress_rate=compress_rate,
            depth=2,
            **kwargs,
        )

        self.local_trans = LocalTrans.builder(
            feature_dim,
            local_window_size=self.trans_window_size,
            depth=depth - 2,
            **kwargs,
        )

    def forward(self, feature):
        feature = feature.permute(0, 2, 1)
        split_feature = torch.split(
            feature, self.local_window_size * self.compress_rate, dim=1
        )
        cache_token = self.cache_token.expand(feature.shape[0], -1, -1)
        feature = torch.cat(
            [
                f
                for fs in split_feature
                for f in (
                    cache_token,
                    fs,
                )
            ],
            dim=1,
        )
        # assert feature[:, self.down_trans_window_size: 2*self.down_trans_window_size, :].equal(
        #     feature.reshape(B, -1, self.down_trans_window_size, C)[:, 1, :, :])
        feature = self.down_trans(feature)
        feature = self.local_trans(feature)
        return feature


class ConvUnit(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, snake_act=True, norm=False, dilation=1, kernel_size=7):
        super().__init__()
        total_pad = (kernel_size - 1) * dilation
        self.dw_conv = Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=total_pad // 2,
            groups=dim,
        )  # depth-wise conv

        self.norm = (
            ChannelNorm(dim, data_format="channels_last") if norm else nn.Identity()
        )
        self.pw_conv1 = Linear(
            dim, 4 * dim
        )  # point-wise/1x1 conv, implemented with linear layer

        if snake_act:
            self.act = Snake1d(4 * dim, data_format="channels_last")
        else:
            self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pw_conv2 = Linear(4 * dim, dim)

    def forward(self, x):
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw_conv2(x)
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        return x


class Residual(nn.Module):
    def __init__(
        self, module: nn.Module, drop_prob: float = 0.0, scale_by_keep: bool = True
    ):
        super().__init__()
        assert 0 <= drop_prob < 1
        self.module = module
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def drop_path(self, x_side: Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x_side
        keep_prob = 1 - self.drop_prob
        shape = (x_side.shape[0],) + (1,) * (x_side.ndim - 1)
        keep_mask = x_side.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep:
            keep_mask.div_(keep_prob)
        return x_side * keep_mask

    def forward(self, x: Tensor):
        x_side = self.module(x)
        x_side = self.drop_path(x_side)
        return x + x_side


ResidualUnit = lambda *args, drop_rate=0.0, **kwargs: Residual(
    ConvUnit(*args, **kwargs), drop_prob=drop_rate
)


class LegacyUnit(nn.Module):
    def __init__(self, dim, snake_act=True, norm=False, dilation=1, kernel_size=7):
        super().__init__()
        assert snake_act, "LegacyUnit only supports snake_act=True"
        assert norm == False, "LegacyUnit only supports norm=False"
        total_pad = (kernel_size - 1) * dilation
        self.block = nn.Sequential(
            Snake1d(dim),
            Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=total_pad // 2,
            ),
            Snake1d(dim),
            Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


ResidualLegacyUnit = lambda *args, **kwargs: Residual(
    LegacyUnit(*args, **kwargs), drop_prob=0.0
)

BaseUnit = ResidualUnit


class Encoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        strides: tuple = (2, 2, 2, 2),
        depths: tuple = (1, 1, 1, 1, 1),
        dims: tuple = (32, 64, 128, 256, 512),
        drop_path_rate: float = 0.0,
        use_norm=False,
        use_snake_act=True,
    ):
        super().__init__()
        # Create first convolution
        blocks = [
            # Conv1d(1, dims[0], kernel_size=7, padding=3),
            FirstBlock(dims[0]),
        ]

        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i_d, o_d, stride, depth in zip(dims[:-1], dims[1:], strides, depths):
            stage = nn.Sequential(
                *[
                    BaseUnit(
                        dim=i_d,
                        drop_rate=drop_path_rates[cur + j],
                        snake_act=use_snake_act,
                        norm=use_norm,
                    )
                    for j in range(depth)
                ]
            )
            down_layer = nn.Sequential(
                Conv1d(i_d, o_d, kernel_size=stride, stride=stride),
                ChannelNorm(o_d, data_format="channels_first")
                if use_norm
                else nn.Identity(),
            )
            blocks += [stage, down_layer]
            cur += depth

        # Create last convolution
        blocks += [
            nn.Sequential(
                *[
                    BaseUnit(
                        dim=dims[-1],
                        drop_rate=drop_path_rates[cur + j],
                        snake_act=use_snake_act,
                        norm=use_norm,
                    )
                    for j in range(depths[-1])
                ]
            ),
            # Snake1d(dims[-1]),
            Conv1d(dims[-1], feature_dim, kernel_size=3, padding=1),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class DistillCodecEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(
            feature_dim=512,
            strides=(4, 4, 4, 4),
            depths=(1, 1, 1, 2),
            dims=(32, 64, 128, 256),
        )
        self.en_encoder = CompressedLocalEncoderWithCache(
            feature_dim=512,
            local_window_size=300,
            compress_rate=5,
            cache_size=0,
            depth=5,
            use_dynamic_pos_bias=True,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.en_encoder(x)
        return x
