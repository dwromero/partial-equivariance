import torch
from torch.nn.utils import weight_norm as w_norm
import numpy as np
from torch import nn
from math import sqrt

# project
import partial_equiv.general as gral


class SIRENBase(torch.nn.Module):
    def __init__(
        self,
        dim_input_space: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        init_scale: float,
        weight_norm: bool,
        bias: bool,
        fix_integer: bool,
        fix_integer_with_bias: bool,
        fix_integer_with_geom: bool,
        omega_0: float,
        learn_omega_0: bool,
        omega_1: float,
        learn_omega_1: bool,
        omega_2: float,
        learn_omega_2: bool,
        Linear_hidden: torch.nn.Module,
        Linear_out: torch.nn.Module,
    ):

        super().__init__()

        # Save params in self
        self.dim_input_space = dim_input_space
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.no_layers = no_layers

        self.init_scale = init_scale
        self.fix_integer = fix_integer
        self.fix_integer_with_bias = fix_integer_with_bias
        self.fix_integer_with_geom = fix_integer_with_geom

        ActivationFunction = gral.nn.Sine

        # Construct the network
        # ---------------------
        # 1st layer:
        kernel_net = [
            Linear_hidden(
                dim_input_space, hidden_channels, bias
            ),
            ActivationFunction(),
        ]

        # Hidden layers:
        for _ in range(no_layers - 2):
            kernel_net.extend(
                [
                    Linear_hidden(
                        hidden_channels,
                        hidden_channels,
                        bias,
                    ),
                    ActivationFunction(),
                ]
            )

        # Last layer:
        kernel_net.extend(
            [
                Linear_out(hidden_channels, out_channels, bias=bias),
            ]
        )
        self.kernel_net = torch.nn.Sequential(*kernel_net)

        # initialize the kernel function
        self.initialize()

        # Weight_norm
        if weight_norm:
            for (i, module) in enumerate(self.kernel_net):
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    # All Conv layers are subclasses of torch.nn.Conv
                    self.kernel_net[i] = w_norm(module)

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

        # omega_1
        if learn_omega_2:
            self.omega_2 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_2.fill_(omega_2)
        else:
            tensor_omega_2 = torch.zeros(1)
            tensor_omega_2.fill_(omega_2)
            self.register_buffer("omega_2", tensor_omega_2)

    def forward(self, x, omegas):
        x_shape = x.shape
        #out = x.clone()
        out = x

        assert len(omegas) == x.shape[1]

        for i, omega in enumerate(omegas):
            if omega == 0:
                out[:, i] *= self.omega_0
            elif omega == 1:
                out[:, i] *= self.omega_1
            elif omega == 2:
                out[:, i] *= self.omega_2
            else:
                print('this should not happen')
                exit(1)

        # Put in_channels dimension at last and compress all other dimensions to one [batch_size, -1, in_channels]
        out = out.view(x_shape[0], x_shape[1], -1).transpose(1, 2)
        # Pass through the network
        out = self.kernel_net(out)
        # Restore shape
        out = out.transpose(1, 2).view(x_shape[0], -1, *x_shape[2:])

        return out

    def initialize(self):
        net_layer = 1
        for (i, m) in enumerate(self.modules()):
            if isinstance(
                m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)
            ):
                # First layer
                if net_layer == 1:
                    w_std = 1 / m.weight.shape[1]
                    m.weight.data.uniform_(-w_std, w_std)

                    if self.fix_integer or self.fix_integer_with_bias:
                        if m.weight.shape[1] == 6:
                            print('fixed 6')
                            m.weight.data[:, 2:4] = m.weight.data[:, 2:4] * 0.0 + 1.0
                        elif m.weight.shape[1] == 5:
                            print('fixed 5')
                            m.weight.data[:, 2:3] = m.weight.data[:, 2:3] * 0.0 + 1.0
                        elif m.weight.shape[1] == 4:
                            print('fixed 4')
                            m.weight.data[:, 2:4] = m.weight.data[:, 2:4] * 0.0 + 1.0
                        elif m.weight.shape[1] == 3:
                            print('fixed 3')
                            m.weight.data[:, 2:3] = m.weight.data[:, 2:3] * 0.0 + 1.0
                        else:
                            print('no fix')

                    if type(self.fix_integer_with_geom) == float:
                        print('FIX WITH GEOM!')
                        p = self.fix_integer_with_geom
                        if m.weight.shape[1] == 6:
                            print('fixed 6')
                            m.weight.data[:, 2:4] = m.weight.data[:, 2:4] * 0.0 + np.random.geometric(p, size=m.weight.data[:, 2:4].shape)
                        elif m.weight.shape[1] == 5:
                            print('fixed 5')
                            m.weight.data[:, 2:3] = m.weight.data[:, 2:3] * 0.0 + np.random.geometric(p, size=m.weight.data[:, 2:3].shape)
                        elif m.weight.shape[1] == 4:
                            print('fixed 4')
                            m.weight.data[:, 2:4] = m.weight.data[:, 2:4] * 0.0 + np.random.geometric(p, size=m.weight.data[:, 2:4].shape)
                        elif m.weight.shape[1] == 3:
                            print('fixed 3')
                            m.weight.data[:, 2:3] = m.weight.data[:, 2:3] * 0.0 + np.random.geometric(p, size=m.weight.data[:, 2:3].shape)
                        else:
                            print('no fix')
                        print(np.unique(m.weight.data.detach().cpu().numpy(), return_counts=True))

                else:
                    w_std = sqrt(6.0 / m.weight.shape[1]) * self.init_scale
                    m.weight.data.uniform_(
                        -w_std,
                        w_std,
                    )
                net_layer += 1
                # Important! Bias is not defined in original SIREN implementation
                if m.bias is not None:
                    if self.fix_integer_with_bias or (type(self.fix_integer_with_geom) == float):
                        print('fixed with bias')
                        if m.weight.shape[1] == 6:
                            m.bias.data[2:4].uniform_(-1.0, 1.0)
                            print('bias', m.bias.data.shape, '2:4')
                        elif m.weight.shape[1] == 5:
                            m.bias.data[2:3].uniform_(-1.0, 1.0)
                            print('bias', m.bias.data.shape, '2:3')
                        elif m.weight.shape[1] == 4:
                            m.bias.data[2:4].uniform_(-1.0, 1.0)
                            print('bias', m.bias.data.shape, '2:4')
                        elif m.weight.shape[1] == 3:
                            m.bias.data[2:3].uniform_(-1.0, 1.0)
                            print('bias', m.bias.data.shape, '2:3')
                        else:
                            print('no fix')
                    else:
                        m.bias.data.zero_()


#############################################
#       SIREN as in Romero et al., 2021
##############################################
class SIREN(SIRENBase):
    """SIREN model.
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """

    def __init__(
        self,
        dim_linear: int,
        dim_input_space: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        init_scale: float,
        weight_norm: bool,
        bias: bool,
        fix_integer: bool,
        fix_integer_with_bias: bool,
        fix_integer_with_geom,
        omega_0: float,
        learn_omega_0: bool,
        omega_1: float,
        learn_omega_1: bool,
        omega_2: float,
        learn_omega_2: bool,
    ):

        # There are no native implementations of ConvNd layers, with N > 3. In this case, we must define
        # Linear layers and perform permutations to achieve an equivalent point-wise conv in high dimensions.
        Linear_hidden = globals()["SIRENLayerNd"]
        Linear_out = torch.nn.Linear

        super().__init__(
            dim_input_space=dim_input_space,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            weight_norm=weight_norm,
            no_layers=no_layers,
            init_scale=init_scale,
            bias=bias,
            fix_integer=fix_integer,
            fix_integer_with_bias=fix_integer_with_bias,
            fix_integer_with_geom=fix_integer_with_geom,
            omega_0=omega_0,
            learn_omega_0=learn_omega_0,
            omega_1=omega_1,
            learn_omega_1=learn_omega_1,
            omega_2=omega_2,
            learn_omega_2=learn_omega_2,
            Linear_hidden=Linear_hidden,
            Linear_out=Linear_out,
        )
        self.dim_linear = dim_linear


class SIRENLayerNd(torch.nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * W x + b, where x is >3 dimensional
        """
        super().__init__(
            in_features=in_channels,
            out_features=out_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

