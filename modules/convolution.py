import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Type

from torch.nn.modules import batchnorm

from utils.padding import get_padding


class Conv2d(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple, str] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        activation: str = None,
        batchnorm: bool = False,
        padding_mode = 'zeros'
    ):
        super().__init__()

        if isinstance(padding, str):
            padding = get_padding(padding)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )

        self.activation = getattr(F, activation) if activation else nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.activation(self.bn(self.conv(x)))


class SeparableConv2d(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        bias: bool = True,
        activation: str = None,
        batchnorm: bool = False,
        padding_mode = 'zeros'
    ):
        super().__init__()

        if isinstance(padding, str):
            padding = get_padding(padding)

        self.conv_depthwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size,
            padding=padding, 
            dilation=dilation, 
            groups=in_channels, 
            bias=bias, 
            padding_mode=padding_mode
        )

        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self.activation = getattr(F, activation) if activation else nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.conv_depthwise(x)
        x = self.conv1x1(x)
        return self.activation(self.bn(x))


class Bottleneck2d(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        shrinkage_factor: int,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        bias: bool = True,
        activation: str = None,
        batchnorm: bool = False,
        padding_mode = 'zeros'
    ):
        super().__init__()

        if isinstance(padding, str):
            padding = get_padding(padding)

        self.conv_depthwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size,
            padding=padding, 
            dilation=dilation, 
            groups=in_channels, 
            bias=bias, 
            padding_mode=padding_mode
        )

        self.conv1x1_shrink = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // shrinkage_factor,
            kernel_size=1
        )

        self.conv1x1_expand = nn.Conv2d(
            in_channels=in_channels // shrinkage_factor,
            out_channels=out_channels,
            kernel_size=1
        )

        self.activation = getattr(F, activation) if activation else nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.activation(self.conv_depthwise(x))
        x = self.conv1x1_shrink(x)
        x = self.activation(self.conv1x1_expand(x))
        return self.bn(x)


class ConvResidualBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        num_convs: int,
        kernel_size: int,
        activation: str = None,
        dilation: list = None,
        pool_factor: int = 1,
        pool_type: str = 'max',
        batchnorm: bool = True,
        conv_type: Type = Conv2d
    ):
        super().__init__()   

        self.stem = nn.Sequential([
            conv_type(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding='same',
                dilation=dilation,
                activation=activation,
                batchnorm=batchnorm
            )
            for i in range(num_convs)
        ])

        if pool_factor > 1:
            self.pool = nn.AvgPool2d(pool_factor) if pool_type == 'max' else nn.MaxPool2d()
        else:
            self.pool = nn.Identity()

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.pool(self.skip(x) + self.stem(x))

class BottleneckResidualBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        shrinkage_factor: int, 
        num_convs: int,
        kernel_size: int,
        activation: str = None,
        dilation: list = None,
        pool_factor: int = 1,
    ):
        super().__init__()   

        self.stem = nn.Sequential([
            Bottleneck2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                shrinkage_factor=shrinkage_factor,
                padding='same',
                dilation=dilation,
                activation=activation,
                batchnorm=True
            )
            for i in range(num_convs)
        ])

        self.pool = nn.AvgPool2d(pool_factor) if pool_factor > 1 else nn.Identity()

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.pool(self.skip(x) + self.stem(x))
         