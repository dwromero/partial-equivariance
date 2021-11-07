import torch
from partial_equiv.partial_gconv.conv import LiftingConv, GroupConv, PointwiseGroupConv
from torch.nn import Conv1d, Conv2d, Conv3d

# typing
from typing import Union


class LnLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """
        Computes the Ln loss on the CKConv kernels in a CKCNN.
        :param weight_loss: Specifies the weight with which the loss will be summed to the total loss.
        :param norm_type: Type of norm, e.g., 1 = L1 loss, 2 = L2 loss, ...
        """
        super().__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def forward(
        self,
        model: torch.nn.Module,
    ):
        loss = 0.0
        # Go through modules that are instances of CKConvs
        for m in model.modules():
            if isinstance(m, (LiftingConv, GroupConv, PointwiseGroupConv)):
                loss += m.conv_kernel.norm(self.norm_type)

                if m.bias is not None:
                    loss += m.bias.norm(self.norm_type)

            # if isinstance(m, (Conv1d, Conv2d, Conv3d)):
            #     loss += m.weight.norm(self.norm_type)
            #     loss += m.bias.norm(self.norm_type)

        loss = self.weight_loss * loss
        return loss
