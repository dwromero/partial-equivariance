import torch

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    # append source
    import os
    import sys

    partial_equiv_source = os.path.join(os.getcwd(), "..")
    if partial_equiv_source not in sys.path:
        sys.path.append(partial_equiv_source)

    from partial_equiv.partial_gconv.conv import LiftingConv, GroupConv
    import partial_equiv.groups as groups

    device = "cpu"
    group = groups.SE2()

    liftconv = LiftingConv(
        in_channels=3,
        out_channels=20,
        group=group,
        no_lifting_samples=8,
        group_sampling_method=groups.SamplingMethods.DETERMINISTIC,
        kernel_size="7",
        kernel_type="Gabor",
        kernelnet_hidden_channels=32,
        kernelnet_no_layers=3,
        bias=True,
    )

    gconv = GroupConv(
        in_channels=20,
        out_channels=5,
        group=group,
        no_group_samples=4,
        group_sampling_method=groups.SamplingMethods.DETERMINISTIC,
        kernel_size="7",
        kernel_type="Gabor",
        kernelnet_hidden_channels=32,
        kernelnet_no_layers=3,
        bias=True,
    )

    x = torch.rand([50, 3, 21, 21], device=device)

    liftconv = liftconv.to(device)
    gconv = gconv.to(device)

    out_lift, g_elems = liftconv(x)
    out_group = gconv(out_lift, g_elems)

    print("here")
