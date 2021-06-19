from modules.attention import ResidualAttentionBlock
from modules.involution import Involution2d
import torch
import torch.nn as nn

import modules


class GenericBackbone(nn.Module):

    def __init__(
        self,
        module_configs : list[dict]
    ):
        '''
        args:
            module_configs: list[dict]
                list of parameters for each module
                expected format ("type" : str, "kwargs" : dict)
        '''
        super().__init__()

        self.backbone = nn.Sequential(
            *[getattr(modules, module_config["type"])(**module_config["kwargs"]) for module_config in module_configs]
        )

    def forward(self, x: torch.Tensor):
        return self.backbone(x)
