"""Small ResNet with policy and ownership heads.

Input:  [B, N_PLANES, size, size]   feature tensor.
Output: policy_logits [B, size*size + 1]
        ownership     [B, size, size]   in (-1, 1) via tanh.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphabetago.features import N_PLANES


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyOwnershipNet(nn.Module):
    def __init__(
        self,
        board_size: int = 9,
        in_planes: int = N_PLANES,
        channels: int = 32,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.board_size = board_size
        self.in_planes = in_planes
        self.channels = channels
        self.n_blocks = n_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        # Policy head: 2-channel reduction, then a linear over flattened map.
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # Ownership head: 1-channel reduction, tanh output per board point.
        self.ownership_conv = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        o = self.ownership_conv(x).squeeze(1)
        ownership = torch.tanh(o)

        return policy_logits, ownership

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
