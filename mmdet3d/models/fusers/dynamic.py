from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["DynamicFuser"]


class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)


@FUSERS.register_module()
class DynamicFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_concate = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
        self.se_block = SE_Block(out_channels)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        ct = self.channel_concate(torch.cat(inputs, dim=1))
        return self.se(ct)
