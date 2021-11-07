"""
This file implements a simple group CNN as in Cohen & Welling 2016. The difference is that here the filters are
parameterized as MLPs on the group, and thus can be sampled at arbitrary angles.
"""
# torch
import copy
import torch

# other
from functools import partial

# project
import partial_equiv.partial_gconv as partial_gconv
import partial_equiv.general as gral
from partial_equiv.general.nn import ApplyFirstElem

# typing
from typing import Tuple, Union
from partial_equiv.groups import Group
from omegaconf import OmegaConf


class ConvBNActDropoutBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        dropout: float,
        NormType: torch.nn.Module,
    ):
        super().__init__()

        # Conv layer
        self.conv = partial_gconv.GroupConv(
            in_channels=in_channels,
            out_channels=out_channels,
            group=copy.deepcopy(group),
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

        # Dropout layer
        self.dp = ApplyFirstElem(torch.nn.Dropout(dropout))

        # Normalization layer
        self.norm = ApplyFirstElem(NormType(out_channels))

        # Activation
        self.activ = ApplyFirstElem(torch.nn.ReLU())

    def forward(self, input_tuple):
        return self.dp(self.activ(self.norm(self.conv(input_tuple))))


class GCNN(torch.nn.Module):
    """
    Group Equivariant Network as in Cohen & Welling, 2016.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_group: Group,
        net_config: OmegaConf,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        **kwargs,
    ):
        super().__init__()

        # Unpack arguments from net_config
        hidden_channels = net_config.no_hidden
        norm = net_config.norm
        no_blocks = net_config.no_blocks
        dropout = net_config.dropout
        pool_blocks = net_config.pool_blocks

        # Params in self
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Define type of normalization layer to use
        if norm == "BatchNorm":
            NormType = getattr(torch.nn, f"BatchNorm{base_group.dimension}d")
        elif norm == "LayerNorm":
            NormType = gral.nn.LayerNorm
        else:
            raise NotImplementedError(f"No norm type {norm} found.")

        # Lifting
        self.lift_conv = partial_gconv.LiftingConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            group=copy.deepcopy(base_group),
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )
        # Lifting Dropout layer
        self.lift_dropout = ApplyFirstElem(torch.nn.Dropout(dropout))
        # Lifting normalization layer
        self.lift_norm = ApplyFirstElem(NormType(hidden_channels))

        # Group layers
        blocks = []
        for i in range(no_blocks):

            blocks.append(
                ConvBNActDropoutBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    group=base_group,
                    base_group_config=base_group_config,
                    kernel_config=kernel_config,
                    conv_config=conv_config,
                    dropout=dropout,
                    NormType=NormType,
                )
            )

            # Pool layer
            if (i + 1) in pool_blocks:
                blocks.append(
                    ApplyFirstElem(
                        partial_gconv.pool.MaxPoolRn(
                            kernel_size=2,
                            stride=2,
                            padding=0,
                        )
                    )
                )

        self.blocks = torch.nn.Sequential(*blocks)

        # Last layer
        self.last_conv = partial_gconv.GroupConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            group=copy.deepcopy(base_group),
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

        # Activation layer
        self.activ = ApplyFirstElem(torch.nn.ReLU())

    def forward(self, x):
        out, g_samples = self.lift_dropout(
            self.activ(self.lift_norm(self.lift_conv(x)))
        )
        out, g_samples = self.blocks([out, g_samples])
        out, _ = self.last_conv([out, g_samples])
        out = torch.mean(out, dim=(-1, -2), keepdim=True)
        return torch.max(out, dim=-3)[0].view(-1, self.out_channels)
