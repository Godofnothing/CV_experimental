import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Union

class ResidualAttentionBlock(nn.Module):

    def __init__(
        self, 
        embed_dim: int, 
        feedforward_dim: int,
        n_heads: int, 
        mask: torch.Tensor = None,
        activation: str = 'GELU',
        dropout: float=0.0
    ):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)

        self.ln_1 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(OrderedDict([
            ("fc_in", nn.Linear(embed_dim, feedforward_dim)),
            ("activation", getattr(nn, activation)),
            ("fc_out", nn.Linear(feedforward_dim, embed_dim))
        ]))

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mask = mask

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.ln_1(x)
        x = identity + self.multihead_attn(x, x, x, need_weights=False, attn_mask=self.mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class VisualTransformer(nn.Module):

    def __init__(
        self, 
        input_size: Union[int, tuple],
        embed_dim: int, 
        feedforward_dim: int,
        patch_size: Union[int, tuple],
        n_layers: int,
        n_heads: int,
        dropout: float = 0.0,
        activation: str = 'GELU',
        create_cls_token: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.create_cls_token = create_cls_token

        input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.n_patches = input_size[0] * input_size[1] // (patch_size[0] * patch_size[1])

        scale = embed_dim ** -0.5

        self.create_patches = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size, 
            bias=False
        )

        self.ln_pre = nn.LayerNorm(normalized_shape=(embed_dim, self.n_patches + self.create_cls_token))
        self.ln_out = nn.LayerNorm(normalized_shape=(embed_dim, self.n_patches + self.create_cls_token))

        self.encoder = nn.Sequential(
            *[ResidualAttentionBlock(embed_dim, feedforward_dim, n_heads, dropout=dropout, activation=activation, dropout=dropout) 
            for _ in range(n_layers)] 
        )

        self.positional_embedding = nn.Parameter(
            scale * torch.randn(embed_dim, self.n_patches + self.create_cls_token)
        )

        if self.create_cls_token:
            self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))

    def forward(self, x: torch.Tensor):
        # patches (B C n_x n_y)
        patches = self.create_patches(x)
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)

        if self.create_cls_token:
            # broadcast class_embeddings to the input shape
            class_embeddings = self.class_embedding[None, :, None].repeat(patches.shape[0], 1, 1)
            # concatenate CLS token and patches
            patches = torch.cat([class_embeddings, patches], dim=2)

        # add positional encoding
        patches = patches + self.positional_embedding
        # main branch
        patches = self.ln_pre(patches)
        # (B C L) -> (L B C)
        patches = patches.permute(2, 0, 1) 
        patches = self.encoder(patches)
        # (L B C) -> (B C L)
        patches = patches.permute(1, 2, 0) 
        patches = self.ln_out(patches)

        return patches
    