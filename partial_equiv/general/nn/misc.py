import torch


# From LieConv
class Expression(torch.nn.Module):
    def __init__(self, func):
        """
        Creates a torch.nn.Module that applies the function func.
        :param func: lambda function
        """
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def Multiply(
    omega_0: float,
):
    """
    out = omega_0 * x
    """
    return Expression(lambda x: omega_0 * x)
