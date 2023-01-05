from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import FUSERS

__all__ = ["FlowFuser"]


@FUSERS.register_module()
class FlowFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.transform = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, 2, 3, padding=1, bias=False),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        camera_feat, lidar_feat = inputs
        flow_field = self.transform(torch.cat(inputs, dim=1))
        h, w = flow_field.shape[2:]
        warpped_feature = self.flow_warp(camera_feat, flow_field, size=(h, w))

        return torch.cat([warpped_feature, lidar_feat], dim=1)

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output