from typing import Optional

import torch
import matplotlib.pyplot as plt

font_size = 14


def img_show(img: torch.Tensor, title: Optional[str] = ""):
    plt.imshow(img[0].detach(), cmap="gray", vmin=0, vmax=1)
    plt.title(title, fontsize=font_size)
    plt.show()


def img_two_show(img1: torch.Tensor, title1: str, img2: torch.Tensor, title2: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    axes[0].imshow(img1[0].detach(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(title1, fontsize=font_size)

    axes[1].imshow(img2[0].detach(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(title2, fontsize=font_size)

    plt.show()
