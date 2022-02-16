# Taken from FlexConvs (Romero et al., 2021b )
from torchvision import datasets, transforms

import os
import os.path
from hydra import utils

VALIDATION_SPLIT = [45000, 5000]


class CIFAR10(datasets.CIFAR10):
    def __init__(
        self,
        partition: str,
        augment: str,
        **kwargs,
    ):
        if "root" in kwargs:
            root = kwargs["root"]
        else:
            root = utils.get_original_cwd()

        import numpy as np
        root = os.path.join(root, f"data_{np.random.randint(999999999999)}")

        transform = []
        if augment == "resnet":
            transform.extend(augmentations_resnet())

        transform.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        transform = transforms.Compose(transform)

        if partition == "train":
            train = True
        elif partition == "test":
            train = False
        else:
            raise NotImplementedError(
                "The dataset partition {} does not exist".format(partition)
            )

        super().__init__(root=root, train=train, transform=transform, download=True)


def augmentations_resnet():
    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
    ]
    return augmentations
