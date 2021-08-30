import torch
import torch.nn as nn
from torch.nn.modules import batchnorm

from typing import Union, Type, Tuple

from .attention import ResidualAttentionBlock
from .convolution import Conv2d, ConvResidualBlock

from utils.patches import images_to_patches, patches_to_images
from utils.padding import get_padding

class AttentionConvMixer(nn.Module):

    def __init__(
        self, 
        n_attention_layers: int,
        in_channels: int,
        out_channels: int,
        feedforward_dim: int,
        n_heads: int,
        input_size:  Union[int, Tuple[int]],
        kernel_size: Union[int, Tuple[int]],
        patch_size: Union[int, Tuple[int]],
        activation_attn: str = 'GELU',
        activation_conv: str = 'relu',
        dropout: float = 0.0,
        conv_type: Type = Conv2d
    ):
        super().__init__()

        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        embed_dim = in_channels * self.patch_size[0] * self.patch_size[1]
        scale = embed_dim ** -0.5

        input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.n_patches = input_size[0] * input_size[1] // (patch_size[0] * patch_size[1])

        self.convs = nn.ModuleList([
            conv_type(
                in_channels=in_channels, 
                out_channels=in_channels, 
                kernel_size=kernel_size, 
                activation=activation_conv,
                batchnorm=True,
                padding=get_padding(kernel_size, mode='same')
            ) for _ in range(n_attention_layers)
        ])

        self.attn_blocks = nn.ModuleList([
            ResidualAttentionBlock(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                n_heads=n_heads,
                activation=activation_attn,
                dropout=dropout
            ) for _ in range(n_attention_layers)
        ])

        self.final_conv = conv_type(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            activation=activation_conv,
            batchnorm=True,
            padding=get_padding(kernel_size, mode='same')
        )

        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.n_patches, embed_dim)
        )

    def forward(self, x: torch.Tensor):
        '''
        args:
            x: torch.Tensor of shape (B, C, H, W)
        '''
        h_, w_ = x.shape[-2:]

        for idx, (conv_, attn_) in enumerate(zip(self.convs, self.attn_blocks)):
            x = conv_(x)
            # images to patches and then permute (B C L)->(L B C)
            # print(x.shape)
            x = images_to_patches(x, self.patch_size).permute(2, 0, 1)
            if idx == 0:
                x += self.positional_embedding.unsqueeze(1)
            # print(x.shape)
            x = attn_(x)
            # print(x.shape)
            # permute (L B C)->(B C L) and then patches to images
            x = patches_to_images(x.permute(1, 2, 0), (h_, w_), self.patch_size)

        return self.final_conv(x)
