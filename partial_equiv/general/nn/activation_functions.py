# Taken from https://github.com/mfinzi/LieConv

# torch
import torch
from partial_equiv.general.nn.misc import FunctionAsModule


def Swish():
    """x * sigmoid(x)"""
    return FunctionAsModule(lambda x: x * torch.sigmoid(x))


def Sine():
    return FunctionAsModule(lambda x: torch.sin(x))
