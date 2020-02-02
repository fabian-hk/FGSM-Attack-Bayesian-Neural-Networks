from typing import Optional
from pathlib import Path

import torch
import torchvision


def create_folder() -> Path:
    data_path = Path("data/data_set/")
    data_path.mkdir(exist_ok=True, parents=True)
    return data_path


def get_train_loader(batch_size: Optional[int] = 128):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            create_folder(),
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )


def get_test_loader(batch_size: Optional[int] = 128):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            create_folder(),
            train=False,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
