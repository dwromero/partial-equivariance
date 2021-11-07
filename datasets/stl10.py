from torchvision import datasets, transforms

import os
import os.path
from hydra import utils


class STL10(datasets.STL10):
    def __init__(
        self,
        partition: str,
        augment: str,
        **kwargs,
    ):
        assert partition in ["train", "test"]

        if "root" in kwargs:
            root = kwargs["root"]
        else:
            root = utils.get_original_cwd()
            root = os.path.join(root, "data")

        transform = []
        if augment == "resnet":
            transform.extend(augmentations_resnet())

        transform.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
                ),
            ]
        )

        transform = transforms.Compose(transform)

        super().__init__(root=root, split=partition, transform=transform, download=True)


def augmentations_resnet():
    augmentations = [
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
    ]
    return augmentations
