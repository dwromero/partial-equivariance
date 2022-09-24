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


class ConvNormNonlin(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
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

        # Normalization layer
        self.norm = ApplyFirstElem(NormType(out_channels))

        # Activation
        self.activ = ApplyFirstElem(torch.nn.ReLU())

    def forward(self, input_tuple):
        return self.activ(self.norm(self.conv(input_tuple)))


class GCNN(torch.nn.Module):

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
        dropout_blocks = net_config.dropout_blocks
        pool_blocks = net_config.pool_blocks
        block_width_factors = net_config.block_width_factors

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
        self.lift_norm = ApplyFirstElem(NormType(hidden_channels))
        self.lift_nonlinear = ApplyFirstElem(torch.nn.ReLU())

        # Define blocks
        # Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks
                for factor, n_blcks in gral.utils.pairwise_iterable(block_width_factors)
            ]
            width_factors = [
                factor for factor_tuple in width_factors for factor in factor_tuple
            ]

        if len(width_factors) != no_blocks:
            raise ValueError(
                "The size of the width_factors does not matched the number of blocks in the network."
            )

        blocks = []
        for i in range(no_blocks):
            print(f"Block {i + 1}/{no_blocks}")

            if i == 0:
                input_ch = hidden_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                ConvNormNonlin(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    group=base_group,
                    base_group_config=base_group_config,
                    kernel_config=kernel_config,
                    conv_config=conv_config,
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

            # Pool layer
            if (i + 1) in dropout_blocks:
                blocks.append(ApplyFirstElem(torch.nn.Dropout2d(dropout)))

        self.blocks = torch.nn.Sequential(*blocks)

        # Last layer
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        self.last_layer = torch.nn.Linear(in_features=final_no_hidden, out_features=out_channels)

        # # Last layer
        # self.last_conv = partial_gconv.GroupConv(
        #     in_channels=hidden_channels,
        #     out_channels=out_channels,
        #     group=copy.deepcopy(base_group),
        #     base_group_config=base_group_config,
        #     kernel_config=kernel_config,
        #     conv_config=conv_config,
        # )

    def forward(self, x):
        out, g_samples = self.lift_nonlinear(self.lift_norm(self.lift_conv(x)))
        out, g_samples = self.blocks([out, g_samples])
        out = torch.mean(out, dim=(-1, -2, -3))
        out = self.last_layer(out)
        return out
