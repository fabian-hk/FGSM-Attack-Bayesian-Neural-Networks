from pathlib import Path

import torch
import torchvision

data_path = Path("data_set/")
data_path.mkdir(exist_ok=True, parents=False)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        data_path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]),
    ),
    batch_size=128,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        data_path,
        train=False,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]),
    ),
    batch_size=128,
    shuffle=True,
)
