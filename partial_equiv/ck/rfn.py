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
        init_scale: float,
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

        self.init_scale = init_scale

        # Construct the network
        # ---------------------
        self.first_layer = nn.Linear(10, hidden_channels // 2, bias=False)

        self.act = nn.ReLU(inplace=True)

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


    def forward(self, x, N_omega0):
        assert x.size(1) == self.dim_input_space, f"Dim linear should equal dimension of input x.size(1)"
        x_shape = x.shape

        out = x.clone()

        # Apply omega's on inputs
        if N_omega0 > 0:
            out[:, :N_omega0] *= self.omega_0
            out[:, N_omega0:] *= self.omega_1
        else:
            out *= self.omega_0

        # Put in_channels dimension at last and compress all other dimensions to one [batch_size, -1, in_channels]
        out = out.view(x_shape[0], x_shape[1], -1).transpose(1, 2) 

        # # Apply omega on weights
        # W = self.first_layer.weight.clone()
        # if self.dim_input_space in [1, 2, 3]:
        #     W *= self.omega_0
        # elif self.dim_input_space == 4:
        #     W[:, :2] *= self.omega_0
        #     W[:, 2:] *= self.omega_1
        # elif self.dim_input_space == 5:
        #     W[:, :3] *= self.omega_0
        #     W[:, 3:] *= self.omega_1
        # else:
        #     raise NotImplementedError(f"Unknown input space: {self.dim_input_space}")
        # out = torch.einsum('abc,dc->abd', out, W) + self.first_layer.bias.view(1, 1, -1)

        out = self.first_layer(out)

        # Forward-pass through kernel net
        out = torch.cat([torch.cos(out), torch.sin(out)], dim=2)

        for layer in self.mid_layers:
            out = layer(out)
            out = self.act(out)

        out = self.last_layer(out)

        # Restore shape
        out = out.transpose(1, 2).view(x_shape[0], -1, *x_shape[2:])

        return out


    def initialize(self):
        self.first_layer.weight.data.normal_(0.0, 2 * math.pi)
        if self.first_layer.bias:
            self.first_layer.bias.data.zero_()
        self.first_layer.weight.data = self.first_layer.weight.data[:, :self.dim_input_space]

        for layer in self.mid_layers:
            layer.weight.data.normal_(0.0, np.sqrt(2 / self.hidden_channels))
            layer.bias.data.zero_()

        self.last_layer.weight.data.normal_(0.0, np.sqrt(2 / self.hidden_channels))
        self.last_layer.bias.data.zero_()

