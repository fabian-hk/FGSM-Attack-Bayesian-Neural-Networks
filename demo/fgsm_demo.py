import torch

import nn_adversary
from networks import Network
from helper.config import Configuration
from helper.data_loader import get_test_loader
from helper import utils

config = Configuration()

image_id = 1362

nn = Network()
nn.load_model()

test_loader = get_test_loader(1, shuffle=False)
x, y = test_loader.dataset[image_id]

y = torch.tensor([y])

nn_d, nn_imgs, nn_pertubation_imgs = nn_adversary.run_attack(
    nn, x, y, config.epsilons, 3
)

id = 2

utils.img_show(x, f"Image, NN Prediction: {nn_d.iloc[0]['y_']}")
utils.img_show(
    nn_pertubation_imgs[id].cpu(), f"Noise (epsilon: {nn_d.iloc[id]['epsilon']})"
)
utils.img_show(
    nn_imgs[id].cpu(), f"Noise added to image, Prediction: {nn_d.iloc[id]['y_']}"
)
