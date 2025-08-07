import torch
import torch.nn as nn
from einops import rearrange
from torchtune.modules import RotaryPositionalEmbeddings


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        r"""https://github.com/meta-llama/llama/blob/main/llama/model.py"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.weight
        return output


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, rotary_embed: RotaryPositionalEmbeddings
    ):
        super().__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.dim = dim
        self.rotary_embed = rotary_embed

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        assert self.flash, "Must have flash attention."

        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        r"""
        Args:
            x: (b, t, h*d)

        Constants:
            b: batch_size
            t: time steps
            r: 3
            h: heads_num
            d: heads_dim
        """
        B, T, C = x.size()

        q, k, v = rearrange(
            self.c_attn(x), "b t (r h d) -> r b h t d", r=3, h=self.n_heads
        )
        # q, k, v: (b, h, t, d)

        q = self.rotary_embed(q)
        k = self.rotary_embed(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0, is_causal=False
            )

        y = rearrange(y, "b h t d -> b t (h d)")

        y = self.c_proj(y)
        # shape: (b, t, h*d)

        return y


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, rotary_embed: RotaryPositionalEmbeddings
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.att = Attention(dim=dim, n_heads=n_heads, rotary_embed=rotary_embed)
        self.mlp = MLP(dim=dim)

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


if __name__ == "__main__":
    rotary_embed_128 = RotaryPositionalEmbeddings(dim=128)
    transformer_block = TransformerBlock(
        dim=1024, n_heads=8, rotary_embed=rotary_embed_128
    )
    x = torch.randn(2, 128, 1024)
    y = transformer_block(x)
    print(y.shape)
    c = 1
