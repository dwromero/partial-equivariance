import torch
import numpy as np
from torch import nn
from math import sqrt

# project
import partial_equiv.general as gral


class LRF(torch.nn.Module):
    def __init__(
        self,
        dim_input_space: int,
        out_channels: int,
        hidden_channels: int,
        init_scale: float,
        bias: bool,
        omega_0: float,
        learn_omega_0: bool,
        omega_1: float,
        learn_omega_1: bool,
    ):

        super().__init__()

        # Save params in self
        self.dim_input_space = dim_input_space
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Construct the network
        # ---------------------
        # 1st layer:
        self.first_layer = nn.Linear(dim_input_space, hidden_channels // 2, bias)

        # Last layer:
        self.last_layer = nn.Linear(hidden_channels, out_channels, bias=bias)

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0fill_(omega_0)
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
        self.initialize(init_scale)

    def forward(self, x):
        out = x.clone()

        # Apply omega's on inputs
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
        out = self.first_layer(out)

        x_shape = x.shape
        # Put in_channels dimension at last and compress all other dimensions to one [batch_size, -1, in_channels]
        out = x.contiguous().view(x_shape[0], x_shape[1], -1).transpose(1, 2)

        # Pass through the network
        out_h = self.first_layer(out * self.omega_0)
        out_cos = torch.cos(out_h)
        out_sin = torch.sin(out_h)
        out = torch.cat((out_cos, out_sin), -1)
        norm = torch.sqrt(2 / self.hidden_channels)
        out *= norm

        out = self.last_layer(out)

        # Restore shape
        out = out.transpose(1, 2).view(x_shape[0], -1, *x_shape[2:])

        return out.contiguous()

    def initialize(self, init_scale):
        # First layer
        self.first_layer.weight.data.uniform_(0, 1.0)
        if self.first_layer.bias is not None:
            self.first_layer.bias.data.zero_()

        # last layer
        w_std = init_scale
        self.last_layer.weight.data.uniform_(-w_std, w_std)
        if self.last_layer.bias is not None:
            self.last_layer.bias.data.zero_()

