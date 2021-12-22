import torch
import numpy as np
from torch import nn
from math import sqrt

# project
import partial_equiv.general as gral


class SIREN2(torch.nn.Module):
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

        # Save params in self
        self.dim_input_space = dim_input_space
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.no_layers = no_layers
        self.init_scale = init_scale

        self.first_layer = nn.Linear(
                dim_input_space, hidden_channels, bias
            )

        self.mid_layers = []
        for _ in range(no_layers - 2):
            self.mid_layers.append(
                    nn.Linear(
                        hidden_channels,
                        hidden_channels,
                        bias,
                    )
                )
        self.mid_layers = nn.ModuleList(self.mid_layers)

        # Last layer:
        self.last_layer = nn.Linear(hidden_channels, out_channels, bias=bias)

        # initialize the kernel function
        self.initialize(init_scale)

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

    def forward(self, x):
        out = x.clone()

        # Apply omega's on inputs
        if self.dim_input_space in [1, 2, 3]:
            out = out * self.omega_0
        elif self.dim_input_space == 4:
            out[:, :2] *= self.omega_0
            out[:, 2:] *= self.omega_1
        elif self.dim_input_space == 5:
            out[:, :3] *= self.omega_0
            out[:, 3:] *= self.omega_1
        else:
            raise NotImplementedError(f"Unknown input space: {self.dim_input_space}")

        # Put in_channels dimension at last and compress all other dimensions to one [batch_size, -1, in_channels]
        x_shape = x.shape
        out = out.view(x_shape[0], x_shape[1], -1).transpose(1, 2)

        # Pass through the network
        print(111, out.shape)
        print(222, self.first_layer.weight.shape)
        print(333, self.first_layer.bias.shape)
        out = torch.einsum('ijk,lk->ijl', out, self.first_layer.weight) + self.first_layer.bias.view(1, 1, -1)
        out = torch.sin(out)

        for m in self.mid_layers:
            out = m(out)
            out = torch.sin(out)

        out = self.last_layer(out)

        # Restore shape
        out = out.transpose(1, 2).view(x_shape[0], -1, *x_shape[2:])

        return out

    def initialize(self, init_scale):
        # first
        self.first_layer.weight.data.normal_(0.0, 1.0)
        if self.first_layer.bias is not None:
            self.first_layer.bias.data.zero_()

        # mid
        for m in self.mid_layers:
            w_std = sqrt(6.0 / self.hidden_channels)
            m.weight.data.uniform_(-w_std, w_std)
            if m.bias is not None:
                m.bias.data.zero_()

        # last
        w_std = sqrt(6.0 / self.hidden_channels) * init_scale
        self.last_layer.weight.data.uniform_(
            -w_std,
            w_std,
        )
        if self.last_layer.bias is not None:
            self.last_layer.bias.data.zero_()



