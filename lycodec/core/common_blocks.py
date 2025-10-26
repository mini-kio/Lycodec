"""Common reusable neural network building blocks."""

import torch
import torch.nn as nn


class ConvNormAct2d(nn.Module):
    """Conv2d -> GroupNorm -> SiLU activation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=8,
        bias=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvNormAct1d(nn.Module):
    """Conv1d -> GroupNorm -> SiLU activation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=8,
        bias=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvTransposeNormAct2d(nn.Module):
    """ConvTranspose2d -> GroupNorm -> SiLU activation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        groups=8,
        bias=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)
