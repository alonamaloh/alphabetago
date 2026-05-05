"""Small ResNet with policy and ownership heads.

Normalization is channel-wise LayerNorm per spatial position (the ConvNeXt
pattern): for an `[B, C, H, W]` activation, we permute to `[B, H, W, C]`,
apply `nn.LayerNorm(C)`, and permute back. This keeps train and inference
behavior identical (no running statistics) without flattening the spatial
structure the way a vanilla LayerNorm over `(C, H, W)` would.

Input:  [B, N_PLANES, size, size]   feature tensor.
Output: policy_logits [B, size*size + 1]
        ownership     [B, size, size]   in (-1, 1) via tanh.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphabetago.features import N_PLANES


class ChannelLayerNorm(nn.Module):
    """LayerNorm applied over the channel dim of [B, C, H, W] tensors,
    independently per spatial location."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = ChannelLayerNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = ChannelLayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
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
            ChannelLayerNorm(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        # Policy head: 2-channel reduction, then a linear over flattened map.
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_norm = ChannelLayerNorm(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # Ownership head: 1-channel reduction, tanh output per board point.
        self.ownership_conv = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)

        p = F.relu(self.policy_norm(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        o = self.ownership_conv(x).squeeze(1)
        ownership = torch.tanh(o)

        return policy_logits, ownership

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
