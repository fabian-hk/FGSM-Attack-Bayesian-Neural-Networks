import torch
import torchvision

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/tmp",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]),
    ),
    batch_size=128,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/tmp",
        train=False,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]),
    ),
    batch_size=128,
    shuffle=True,
)
