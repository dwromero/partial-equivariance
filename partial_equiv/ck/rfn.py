import math

import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm as w_norm
import numpy as np
from torch import nn
from math import sqrt

# project
import partial_equiv.general as gral


class RFN(torch.nn.Module):
    def __init__(
        self,
        dim_input_space: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        weight_norm: bool,
        bias: bool,
        omega_0: float,
        learn_omega_0: bool,
        omega_1: float,
        learn_omega_1: bool,
    ):

        super().__init__()
        print(f"RFN: {dim_input_space}, {hidden_channels}, {out_channels} ")

        # Save params in self
        self.dim_input_space = dim_input_space
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.no_layers = no_layers

        # Construct the network
        # ---------------------

        # 1st layer:

        # Trick: initialize with 10 output channels and then remove input channels that are too  many after init
        # so that next layers have the same weight values no matter the amount of input channels of first layer
        self.first_layer = nn.Linear(10, hidden_channels, bias=bias)

        self.act1 = gral.nn.Cos()
        self.act2 = nn.ReLU(inplace=True)

        self.mid_layers = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=bias) for _ in range(self.no_layers - 2)])
        self.last_layer = nn.Linear(hidden_channels, out_channels, bias=bias)

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

        # omega_1
        if learn_omega_1:
            self.omega_1 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_1.fill_(omega_1)
        else:
            tensor_omega_1 = torch.zeros(1)
            tensor_omega_1.fill_(omega_1)
            self.register_buffer("omega_1", tensor_omega_1)

        # initialize the kernel function
        self.initialize()


    def forward(self, x):
        assert x.size(1) == self.dim_input_space, f"Dim linear should equal dimension of input x.size(1)"

        out = x.clone()

        x_shape = out.shape

        # Put in_channels dimension at last and compress all other dimensions to one [batch_size, -1, in_channels]
        out = out.view(x_shape[0], x_shape[1], -1).transpose(1, 2) 

        # Apply omega's
        if self.dim_input_space in [1, 2, 3]:
            out = out * self.omega_0
        elif self.dim_input_space == 4:
            out[:, :, :2] *= self.omega_0
            out[:, :, 2:] *= self.omega_1
        elif self.dim_input_space == 5:
            out[:, :, :3] *= self.omega_0
            out[:, :, 3:] *= self.omega_1
        else:
            raise NotImplementedError(f"Unknown input space: {self.dim_input_space}")

        # Forward-pass through kernel net
        out = self.first_layer(out)
        out = self.act1(out)
        out = out * np.sqrt(2 / self.hidden_channels)

        for layer in self.mid_layers:
            out = layer(out)
            out = self.act2(out)

        out = self.last_layer(out)

        # Restore shape
        out = out.transpose(1, 2).view(x_shape[0], -1, *x_shape[2:])

        return out


    def initialize(self):
        for (i, m) in enumerate(self.modules()):
            if isinstance(
                m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)
            ):
                if m == self.first_layer:
                    print(f'Init first layer')
                    # First layer: we initailize as 'random fourier features' and set input channels to correct number
                    # See explanation of trick earlier
                    self.first_layer.weight.data.normal_(0.0, 1.0)
                    self.first_layer.bias.data.uniform_(0.0, 2 * np.pi)
                    self.first_layer.weight.data = self.first_layer.weight.data[:, :self.dim_input_space]
                else:
                    print(f'Init {m}')
                    w_std = sqrt(6.0 / m.weight.shape[1])
                    m.weight.data.uniform_(
                        -w_std,
                        w_std,
                    )
                    if m.bias is not None:
                        m.bias.data.zero_()

