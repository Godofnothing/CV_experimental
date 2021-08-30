import torch
import torch.nn as nn
import torch.nn.functional as F

from .convolution import *

'''
config_format

"input_conv":
    kwargs for Conv2d
"residual_blocks" : 
    list of kwargs for ConvResidualBlock
"num_classes":
    number of classes in the classification problem
'''

class ResNet(nn.Module):

    def __init__(self, config : dict):
        super().__init__()

        self.input_conv = Conv2d(**config["input_conv"])

        self.residual_blocks = nn.Sequential(
            *[ConvResidualBlock(**block_kw) for block_kw in config['residual_blocks']]
        )

        self.output_channels = config['residual_blocks'][-1]['out_channels']

    def get_output_channels(self):
        return self.output_channels

    def forward(self, images : torch.Tensor):
        input_features = self.input_conv(images)
        output_features = self.residual_blocks(input_features)
        return output_features
