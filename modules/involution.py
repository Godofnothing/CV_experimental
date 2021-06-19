import torch.nn as nn
import torch.nn.functional as F

class Involution2d(nn.Module):

    def __init__(
        self, 
        channels : int, 
        kernel_size : int, 
        stride:int = 1,
        shrinkage_ratio:int =1, 
        groups:int = 1,
        activation: str = 'relu',
        kernel_generation_fn_use_bn: bool = True,
        kernel_generation_fn_kernel_size: int =1
    ):
        super(Involution2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.shrinkage_ratio = shrinkage_ratio
        self.groups = groups
        self.kernel_generation_fn_kernel_size = kernel_generation_fn_kernel_size
        self.kernel_generation_fn_use_bn = kernel_generation_fn_use_bn

        self.conv_inner = nn.Conv2d(
            in_channels=channels,
            out_channels=channels // shrinkage_ratio,
            kernel_size=kernel_generation_fn_kernel_size,
        )

        self.conv_outer = nn.Conv2d(
            in_channels=channels // shrinkage_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=kernel_generation_fn_kernel_size,
        )

        self.activation = getattr(F, activation)

        self.avgpool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

        self.bn = nn.BatchNorm2d(channels // shrinkage_ratio) if kernel_generation_fn_use_bn else nn.Identity()

        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        tmp = self.avgpool(x)
        tmp = self.conv_inner(tmp)
        tmp = self.activation(tmp)
        tmp = self.bn(tmp)
        kernel = self.conv_outer(tmp)

        b_, c_, h_, w_ = kernel.shape
        kernel = kernel.view(b_, self.groups, self.kernel_size**2, h_, w_).unsqueeze(2)
        res = self.unfold(x).view(b_, self.groups, c_ // self.groups, self.kernel_size**2, h_, w_)
        res = (kernel * res).sum(dim=3).view(b_, self.channels, h_, w_)

        return res
        