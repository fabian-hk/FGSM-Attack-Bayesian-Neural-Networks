import matplotlib.pyplot as plt


def img_show(img):
    plt.imshow(img[0].detach(), cmap="gray", vmin=0, vmax=1)
    plt.show()
