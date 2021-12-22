# torch
import copy

import math
import torch
from torch import einsum
# from opt_einsum import contract as einsum
import torch.nn.functional as torch_F

# project
import partial_equiv.general.utils as g_utils
import partial_equiv.ck as ck
from partial_equiv.groups import Group, SamplingMethods

# typing
from typing import Tuple
from omegaconf import OmegaConf


class ConvBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        conv_type: str,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        super().__init__()

        # Unpack values from kernel_config
        kernel_type = kernel_config.type
        kernel_no_hidden = kernel_config.no_hidden
        kernel_no_layers = kernel_config.no_layers
        kernel_init_scale = kernel_config.init_scale
        kernel_weight_norm = kernel_config.weight_norm
        kernel_omega0 = kernel_config.omega0
        kernel_learn_omega0 = kernel_config.learn_omega0
        kernel_omega1 = kernel_config.omega1
        kernel_learn_omega1 = kernel_config.learn_omega1
        kernel_size = kernel_config.size
        kernel_activation=kernel_config.activation
        kernel_norm=kernel_config.norm

        # Unpack values from conv_config
        bias = conv_config.bias
        padding = conv_config.padding
        part_rot = conv_config.partial_equiv
        cond_trans = conv_config.cond_trans

        # Unpack values from group_config and save them in self.
        self.group = group
        self.group_no_samples = base_group_config.no_samples
        self.group_sampling_method = base_group_config.sampling_method
        self.group_sample_per_batch_element = base_group_config.sample_per_batch_element
        self.group_sample_per_layer = base_group_config.sample_per_layer

        # Save parameters in self
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.part_rot = part_rot
        self.cond_trans = cond_trans

        # Get the dim_linear as well as the dim_input_space from the type of convolution.
        cond_dims = 2 if self.cond_trans else 0
        if conv_type == "lifting":
            self.dim_linear = self.group.dimension_Rd + cond_dims
            self.dim_input_space = self.group.dimension_Rd + cond_dims
        elif conv_type == "group":
            self.dim_linear = self.group.dimension + cond_dims
            self.dim_input_space = (
                self.group.dimension_Rd + self.group.dimension_stabilizer + cond_dims
            )
        elif conv_type == "pointwise":
            self.dim_linear = self.group.dimension - self.group.dimension_Rd
            self.dim_input_space = self.group.dimension_stabilizer

        # Create kernel
        if kernel_type == "SIREN":
            self.kernelnet = ck.SIREN(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_no_hidden,
                no_layers=kernel_no_layers,
                init_scale=kernel_init_scale,
                weight_norm=kernel_weight_norm,
                bias=True,
                omega_0=kernel_omega0,
                learn_omega_0=kernel_learn_omega0,
                omega_1=kernel_omega1,
                learn_omega_1=kernel_learn_omega1,
            )
        elif kernel_type == "SIREN2":
            self.kernelnet = ck.SIREN2(
                dim_input_space=self.dim_input_space,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_no_hidden,
                no_layers=kernel_no_layers,
                init_scale=kernel_init_scale,
                bias=True,
                omega_0=kernel_omega0,
                learn_omega_0=kernel_learn_omega0,
                omega_1=kernel_omega1,
                learn_omega_1=kernel_learn_omega1,
            )
        elif kernel_type == "Gabor":
            self.kernelnet = ck.GaborNet(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                hidden_channels=kernel_no_hidden,
                out_channels=out_channels * in_channels,
                no_layers=kernel_no_layers,
                bias=True,
            )
        elif kernel_type == "Fourier":
            self.kernelnet = ck.FourierNet(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                hidden_channels=kernel_no_hidden,
                out_channels=out_channels * in_channels,
                no_layers=kernel_no_layers,
                bias=True,
            )
        elif kernel_type == "MAGNet":
            self.kernelnet = ck.MAGNet(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                hidden_channels=kernel_no_hidden,
                out_channels=out_channels * in_channels,
                no_layers=kernel_no_layers,
                steerable=False,  # TODO: Not implemented in 3D
                bias=True,
            )
        elif kernel_type == "MLP":
            self.kernelnet = ck.MLP(
                dim_linear=self.dim_linear,
                dim_input_space=self.dim_input_space,
                hidden_channels=kernel_no_hidden,
                out_channels=out_channels * in_channels,
                no_layers=kernel_no_layers,
                activation=kernel_activation,
                norm_type=kernel_norm,
                bias=True,
            )
        else:
            raise NotImplementedError(f"kernel_type {kernel_type} not implemented.")

        # Define position holder for relative positions
        self.rel_positions = None

        # Define bias:
        bias = False # TODO: check, is this really the right way to do bias? also the comment says "other dimensions", but is missing in code??
        if bias:
            bias = torch.zeros((1, out_channels))  # [Batch, Ch, other_dimensions]
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None

    def handle_rel_positions_on_Rd(self, x):
        """
        Handles the vector of relative positions on Rd.
        In the case of lifting convolutions, the Rd vector is given to the KernelNet.
        In the case of group convolutions, this vector will **later** be acted upon, and concatenated
        with the additional group coordinates, in order to get the vector of positions given to the KernelNet.
        """
        if self.rel_positions is None:

            kernel_size = torch.zeros(1).int()

            # Decide the extend of the rel_positions vector
            if self.kernel_size == "full":
                kernel_size[0] = (2 * x.shape[-1]) - 1
            elif self.kernel_size == "same":
                kernel_size[0] = x.shape[-1]
            elif int(self.kernel_size) % 2 == 1:
                # Odd number
                kernel_size[0] = int(self.kernel_size)
            else:
                raise ValueError(
                    f'The horizon argument of the operation must be either "full", "same" or an odd number in string format. Current value: {self.kernel_size}'
                )

            # Creates the vector of relative positions.
            rel_positions = g_utils.rel_positions_grid(
                grid_sizes=kernel_size.repeat(self.group.dimension_Rd)
            )
            self.rel_positions = rel_positions.to(x.device)
            # -> With form: [dim, x_dimension, y_dimension]

        return self.rel_positions


class LiftingConv(ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        conv_type = "lifting"
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type,
            group=group,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

        # Construct the probability variables given the group.
        # IMPORTANT! Not used here!
        # probs = self.group.construct_probability_variables(
        #     self.group_sampling_method,
        #     base_group_config.no_samples,
        # )
        self.probs = None

    def forward(self, x):
        """
        :param x: Input: Function on Rd of size [batch_size, in_channels, * ]
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """

        # Get input grid of conv kernel
        input_rel_pos = self.handle_rel_positions_on_Rd(x)

        # Get output grid relative positions, if needed
        if not hasattr(self, 'output_rel_pos'):
            # TODO: now assuming x to have same size always
            self.output_rel_pos = g_utils.rel_positions_grid(
                grid_sizes=x.shape[-2:]
            )
            self.output_rel_pos = self.output_rel_pos.to(x.device)

        output_rel_pos = self.output_rel_pos

        # Define the number of independent samples to take from the group. If self.sample_per_batch_element == True, and
        # self.group_sampling_method == RANDOM, then batch_size independent self.group_no_samples samples from the group
        # will be taken for each batch element. Otherwise, the same rotations are used across batch elements.

        if (
            self.group_sample_per_batch_element
            # and self.group_sampling_method == SamplingMethods.RANDOM
        ):
            no_samples = x.shape[0]  # batch size
        else:
            no_samples = 1

        # Sample no_group_elements from the group, batch_size times.
        g_elems = self.group.sample_from_stabilizer(
            no_samples=no_samples,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            partial_equivariance=False,  # We always lift first to the group.
            probs=self.probs,
        )  # [no_samples, self.group_no_samples]

        # For R2, we don't need to go to the LieAlgebra. We can parameterize the kernel directly on the input space
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(g_elems.view(-1, self.group.dimension_stabilizer)), input_rel_pos
        )

        kernel_size = acted_rel_pos_Rd.shape[-2:]
        image_size = x.shape[-2:]

        if self.cond_trans:
            output_g_no_elems = self.group_no_samples

            acted_rel_pos = torch.cat(
                (
                    # Expand the acted rel pos Rd input_g_no_elems times along the "group axis", and "output group axes"
                    # [no_samples * output_g_no_elems, 2, kernel_size_y, kernel_size_x]
                    # +->  [no_samples * output_g_no_elems, 2, kernel_size_y, kernel_size_x, output_size_y, output_size_x]
                    acted_rel_pos_Rd.contiguous().view(
                        no_samples * output_g_no_elems,
                        2,
                        *kernel_size,
                        1,
                        1)
                    .expand(
                        *(-1,) * 4, *image_size
                    ),
                    # Expand the output rel pos Rd output_g_no_elems times along the "group axis", and "output group axes"
                    # [2, output_size_x, output_size_y]
                    # +->  [no_samples * output_g_no_elems, 2, kernel_size_y, kernel_size_x, output_size_x, output_size_y]
                    output_rel_pos.contiguous().view(
                        1,
                        2,
                        1,
                        1,
                        *image_size)
                    .expand(
                        no_samples * output_g_no_elems,
                        -1,
                        *kernel_size,
                        -1,
                        -1
                    )
                ),
                dim=1,
            )
        else:
            acted_rel_pos = acted_rel_pos_Rd.contiguous()

        self.acted_rel_pos = acted_rel_pos

        # Get the kernel
        conv_kernel = self.kernelnet(acted_rel_pos)

        # if self.cond_trans:
        #     conv_kernel = conv_kernel.view(
        #         no_samples * self.group_no_samples,
        #         self.out_channels,
        #         self.in_channels,
        #         *kernel_size,
        #         *image_size
        #     )
        # else:
        #     conv_kernel = conv_kernel.view(
        #         no_samples * self.group_no_samples,
        #         self.out_channels,
        #         self.in_channels,
        #         *kernel_size,
        #     )


        # Filter values outside the sphere

        # TODO: temporary removed masking, should be applied at most efficient location
        # mask = torch.norm(acted_rel_pos_Rd, dim=1) > 1.0
        # if self.cond_trans:
        #     mask = mask.view(mask.size(0), 1, 1, *kernel_size, 1, 1)
        # else:
        #     mask = mask.view(mask.size(0), 1, 1, *kernel_size)

        # mask = mask.expand_as(conv_kernel)

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        if self.cond_trans:
            conv_kernel = conv_kernel.contiguous().view(
                no_samples * self.group_no_samples * self.out_channels,
                self.in_channels,
                *kernel_size,
                *image_size
            )
        else:
            conv_kernel = conv_kernel.contiguous().view(
                no_samples * self.group_no_samples * self.out_channels,
                self.in_channels,
                *kernel_size
            )

        # Convolution:
        if no_samples == 1:
            inp = x
        else:
            inp = x.view(1, -1, *x.shape[2:]),  # Shift batch_size to input

        # assert self.padding == 'same', f"Only implemented same padding."
        # Compute convolution as a 2D convolution
        padding = tuple([x // 2 for x in kernel_size])
        padding = padding + padding
        if self.padding == 'same':
            #inp_pad = torch_F.pad(inp, padding, 'replicate')
            inp_pad = torch_F.pad(inp, padding, 'constant')
        elif self.padding == 'valid':
            inp_pad = pad
        else:
            raise NotImplementedError(f"Unknown padding [{self.padding}].")

        # unfold (transfer input to patch space: bix' -> bpx)
        inp_unf = torch.nn.functional.unfold(inp_pad, kernel_size)

        if self.cond_trans:
            # apply conditional kernel filter
            k_unf = conv_kernel.view(conv_kernel.size(0), conv_kernel.size(1)*kernel_size[0]*kernel_size[1], image_size[0]*image_size[1])

            out = einsum('bpx,opx->box', inp_unf, k_unf)

            # out = torch.nn.functional.fold(out, (32, 32), (1, 1))# , or equivalent that avoids mem copy:
            # out = out.view(out.size(0), -1, *image_size)
        else:
            # Perform convolution as regular broadcasted matrix multiplication
            k_unf = conv_kernel.view(conv_kernel.size(0), conv_kernel.size(1)*kernel_size[0]*kernel_size[1])

            # or einsum
            out = einsum('bpx,op->box', inp_unf, k_unf)

            print(1, out.shape)
            #out = out.view(out.size(0), -1, *image_size)

        print(2, out.shape)
        out = out.view(
            -1, self.group_no_samples, self.out_channels, *image_size
        )
        print(3, out.shape)
        out = out.transpose(1, 2)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, g_elems


class GroupConv(ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        conv_type = "group"
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type,
            group=group,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

        # Construct the probability variables given the group.
        probs = self.group.construct_probability_variables(
            self.group_sampling_method,
            base_group_config.no_samples,
        )
        if self.part_rot:
            self.probs = torch.nn.Parameter(probs)
        else:
            self.register_buffer("probs", probs)

    def forward(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements for which the feature representation is obtained.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems = input_tuple

        # Get input grid of conv kernel
        input_rel_pos = self.handle_rel_positions_on_Rd(x)

        if self.padding == 'same':
            new_image_size = x.shape[-2:]
        else:
            # HACK
            new_image_size = (2, 2)

        # Get output grid relative positions, if needed
        if not hasattr(self, 'output_rel_pos'):
            # TODO: now assuming x to have same size always
            self.output_rel_pos = g_utils.rel_positions_grid(
                grid_sizes=new_image_size
            )
            self.output_rel_pos = self.output_rel_pos.to(x.device)

        output_rel_pos = self.output_rel_pos


        # Define the number of independent samples to take from the group. If self.sample_per_batch_element == True, and
        # self.group_sampling_method == RANDOM, then batch_size independent self.group_no_samples samples from the group
        # will be taken for each batch element. Otherwise, the same rotations are used across batch elements.
        if (
            self.group_sample_per_batch_element
            and self.group_sampling_method == SamplingMethods.RANDOM
        ):
            no_samples = x.shape[0]  # batch size
        else:
            no_samples = 1

        # If self.group_sample_per_layer == True, and self.group_sampling_method == RANDOM, sample at every layer a
        # different set of rotations. Otherwise, the same group elements are used across the network.
        if (
            self.group_sample_per_layer
            and self.group_sampling_method == SamplingMethods.RANDOM
        ) or self.part_rot:
            # Sample from the group
            g_elems = self.group.sample_from_stabilizer(
                no_samples=no_samples,
                no_elements=self.group_no_samples,
                method=self.group_sampling_method,
                device=x.device,
                partial_equivariance=self.part_rot,
                probs=self.probs,
            )
        else:
            g_elems = input_g_elems

        # Act on the grid of positions:

        # Act along the group dimension
        acted_g_elements = self.group.left_action_on_H(
            self.group.inv(g_elems), input_g_elems
        )

        # Normalize elements to coordinates between -1 and 1
        acted_g_elements = self.group.normalize_g_distance(acted_g_elements).float()
        if self.group.__class__.__name__ == 'SE2':
            acted_g_elements = acted_g_elements.unsqueeze(-1)


        # Act on Rd with the resulting elements
        acted_rel_pos_Rd = self.group.left_action_on_Rd(
            self.group.inv(g_elems.view(-1, self.group.dimension_stabilizer)), input_rel_pos
        )

        # Combine both grids
        # Resulting grid: [no_samples * g_elems, group.dim, self.input_g_elems, kernel_size, kernel_size]
        input_g_no_elems = acted_g_elements.shape[2]
        output_g_no_elems = acted_g_elements.shape[1]
        no_samples = acted_g_elements.shape[0]

        kernel_size = acted_rel_pos_Rd.shape[-2:]
        image_size = x.shape[-2:]

        if self.cond_trans:
            acted_group_rel_pos = torch.cat(
                (
                    # Expand the acted rel pos Rd input_g_no_elems times along the "group axis", and "output group axes"
                    # [no_samples * output_g_no_elems, 2, kernel_size_y, kernel_size_x]
                    # +->  [no_samples * output_g_no_elems, 2, input_no_g_elems, kernel_size_y, kernel_size_x, output_size_y, output_size_x]
                    acted_rel_pos_Rd.contiguous().view(
                        no_samples * output_g_no_elems,
                        2,
                        1,
                        *kernel_size,
                        1,
                        1)
                    .expand(
                        *(-1,) * 2, input_g_no_elems, *(-1,) * 2, *new_image_size
                    ),
                    # Expand the acted g elements along the "spatial axes", and "output group axes"
                    # [no_samples, output_g_no_elems, input_g_no_elems, self.group.dimension_stabilizer]
                    # +->  [no_samples * output_g_no_elems, self.group.dimension_stabilizer, input_no_g_elems, kernel_size_y, kernel_size_x, output_size_y, output_size_x]
                    acted_g_elements.transpose(-1, -2)
                    .contiguous()
                    .view(
                        no_samples * output_g_no_elems,
                        self.group.dimension_stabilizer,
                        input_g_no_elems,
                        1,
                        1,
                        1,
                        1,
                    )
                    .expand(
                        -1,
                        -1,
                        -1,
                        *kernel_size,
                        *new_image_size
                    ),
                    # Expand the output rel pos Rd output_g_no_elems times along the "group axis", and "output group axes"
                    # [2, output_size_x, output_size_y]
                    # +->  [no_samples * output_g_no_elems, 2, input_no_g_elems, kernel_size_y, kernel_size_x, output_size_x, output_size_y]
                    output_rel_pos.contiguous().view(
                        1,
                        2,
                        1,
                        1,
                        1,
                        *new_image_size)
                        .expand(
                        no_samples * output_g_no_elems,
                        -1,
                        input_g_no_elems,
                        *kernel_size,
                        -1,
                        -1)
                ),
                dim=1,
            )
            
        else:
            acted_group_rel_pos = torch.cat(
                (
                    # Expand the acted rel pos Rd input_g_no_elems times along the "group axis".
                    # [no_samples * output_g_no_elems, 2, kernel_size_y, kernel_size_x]
                    # +->  [no_samples * output_g_no_elems, 2, input_no_g_elems, kernel_size_y, kernel_size_x]
                    acted_rel_pos_Rd.unsqueeze(2).expand(
                        *(-1,) * 2, input_g_no_elems, *(-1,) * 2
                    ),
                    # Expand the acted g elements along the "spatial axes".
                    # [no_samples, output_g_no_elems, input_g_no_elems, self.group.dimension_stabilizer]
                    # +->  [no_samples * output_g_no_elems, self.group.dimension_stabilizer, input_no_g_elems, kernel_size_y, kernel_size_x]
                    acted_g_elements.transpose(-1, -2)
                    .contiguous()
                    .view(
                        no_samples * output_g_no_elems,
                        self.group.dimension_stabilizer,
                        input_g_no_elems,
                        1,
                        1,
                    )
                    .expand(
                        -1,
                        -1,
                        -1,
                        *kernel_size
                    ),
                ),
                dim=1,
            )
        self.acted_rel_pos = acted_group_rel_pos

        # Get the kernel
        conv_kernel = self.kernelnet(acted_group_rel_pos)

        # TODO: write masking code at efficient location with generalized group convolution
        # conv_kernel = conv_kernel.view(
        #     no_samples * output_g_no_elems,
        #     self.out_channels,
        #     self.in_channels,
        #     *acted_group_rel_pos.shape[2:]
        # )
        # # Filter values outside the sphere
        # mask = torch.norm(acted_rel_pos_Rd, dim=1) > 1.0
        # if self.cond_trans:
        #     mask = mask.view(mask.size(0), 1, 1, 1, *kernel_size, 1, 1)
        # else:
        #     mask = mask.view(mask.size(0), 1, 1, 1, *kernel_size)
        # mask = mask.expand_as(conv_kernel)

        # conv_kernel[mask] = 0

        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        if self.cond_trans:
            print(f'no_samples: {no_samples}')
            print(f'gr_samples: {self.group_no_samples}')
            print(f'o_channels: {self.out_channels}')
            print(f'i_channels: {self.in_channels}, {input_g_no_elems}')
            print(f'{kernel_size}, {new_image_size}')
            conv_kernel = conv_kernel.contiguous().view(
                no_samples * output_g_no_elems * self.out_channels,
                self.in_channels * input_g_no_elems,
                *kernel_size,
                *new_image_size
            )
        else:
            conv_kernel = conv_kernel.contiguous().view(
                no_samples * output_g_no_elems * self.out_channels,
                self.in_channels * input_g_no_elems,
                *kernel_size
            )

        # Convolution:
        if no_samples == 1:
            inp = x.contiguous().view(-1, self.in_channels * input_g_no_elems, *x.shape[-2:])
        else:
            inp = x.contiguous().view(1, -1, *x.shape[3:]),

        # Compute convolution as a 2D convolution
        padding = tuple([x // 2 for x in kernel_size])
        padding = padding + padding
        if self.padding == 'same':
            inp_pad = torch_F.pad(inp, padding, 'replicate')
        elif self.padding == 'valid':
            inp_pad = inp
        else:
            raise NotImplementedError(f"Unknown padding [{self.padding}].")

        # unfold (transfer input to patch space: bix' -> bpx)
        inp_unf = torch.nn.functional.unfold(inp_pad, kernel_size)

        if self.cond_trans:
            # apply conditional kernel filter
            k_unf = conv_kernel.view(conv_kernel.size(0), conv_kernel.size(1)*kernel_size[0]*kernel_size[1], new_image_size[0]*new_image_size[1])

            if self.kernelnet.omega_1 == 0.0:
                assert (k_unf.std(2) == 0.0).all(), f"No equivariance check"

            out = einsum('bpx,opx->box', inp_unf, k_unf)

            # out = torch.nn.functional.fold(out, (32, 32), (1, 1)), equivalent that avoids mem copy:
            out = out.contiguous().view(out.size(0), -1, *new_image_size)
        else:
            # Perform convolution as regular broadcasted matrix multiplication
            k_unf = conv_kernel.view(conv_kernel.size(0), conv_kernel.size(1)*kernel_size[0]*kernel_size[1])

            # using transpose
            # out = inp_unf.transpose(1, 2).matmul(k_unf.t()).transpose(1, 2)

            # or einsum
            out = einsum('bpx,op->box', inp_unf, k_unf)

            # out = torch.nn.functional.fold(out, (32, 32), (1, 1)) #, equivalent that avoids mem copy:
            out = out.contiguous().view(out.size(0), -1, *new_image_size)

            # # Original conv:
            # out_old = torch_F.conv2d(
            #     input=inp_pad,
            #     weight=conv_kernel,
            #     groups=no_samples,
            # )
            # print(torch.mean(torch.abs(out - out_old)))

        out = out.view(
            -1, output_g_no_elems, self.out_channels, *out.shape[2:]
        ).transpose(1, 2)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, g_elems


class PointwiseGroupConv(ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        kernel_config = copy.deepcopy(kernel_config)
        kernel_config.size = "1"

        conv_config = copy.deepcopy(conv_config)
        conv_config.bias = False
        conv_config.partial_equiv = False

        conv_type = "pointwise"
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type,
            group=group,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

    def forward(
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """
        :param input_tuple: Consists of two elements:
            (1) The input function in the group [batch_size, in_channels, * ]
            (2) The grid of sampled group elements from the input.
            (3) The grid of sampled group elements from the output.
        :return: Output: Function on the group of size [batch_size, out_channels, no_lifting_samples, * ]
        """
        # Unpack input tuple
        x, input_g_elems, output_g_elems = input_tuple

        # Act along the group dimension
        acted_g_elements = self.group.left_action_on_H(
            self.group.inv(output_g_elems), input_g_elems
        )

        # Normalize elements to coordinates between -1 and 1
        acted_g_elements = self.group.normalize_g_distance(acted_g_elements).float()
        if self.group.__class__.__name__ == 'SE2':
            acted_g_elements = acted_g_elements.unsqueeze(-1)

        input_g_no_elems = acted_g_elements.shape[2]
        output_g_no_elems = acted_g_elements.shape[1]
        no_samples = acted_g_elements.shape[0]

        acted_g_elements = (
            acted_g_elements.transpose(-1, -2)
            .contiguous()
            .view(
                no_samples * output_g_no_elems,
                self.group.dimension_stabilizer,
                input_g_no_elems,
            )
        )
        # Resulting grid: [no_samples * g_elems, group.dimension_stabilizer, self.input_g_elems]

        # Get the kernel
        conv_kernel = self.kernelnet(acted_g_elements).view(
            no_samples * output_g_no_elems,
            self.out_channels,
            self.in_channels,
            input_g_no_elems,
            1,
            1,
        )
        self.conv_kernel = conv_kernel

        # Reshape conv_kernel for convolution
        conv_kernel = conv_kernel.contiguous().view(
            no_samples * output_g_no_elems * self.out_channels,
            self.in_channels * input_g_no_elems,
            *conv_kernel.shape[-2:],
        )

        # Convolution:

        if no_samples == 1:
            inp = x.contiguous().view(-1, self.in_channels * input_g_no_elems, *x.shape[-2:])
        else:
            inp = x.contiguous().view(1, -1, *x.shape[3:])

        out = torch_F.conv2d(
            input=inp,
            weight=conv_kernel,
            padding=self.padding,
            groups=no_samples,
        )
        outt = torch.einsum('bcij,dckl->bdij', inp, conv_kernel)

        out = out.view(
            -1, output_g_no_elems, self.out_channels, *out.shape[2:]
        ).transpose(1, 2)

        # Add bias:
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *(1,) * (len(out.shape) - 2))

        return out, output_g_elems
