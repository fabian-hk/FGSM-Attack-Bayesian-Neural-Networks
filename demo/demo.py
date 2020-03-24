import torch
import pyro

import nn_adversary
import bnn_adversary
from networks import Network, BNNWrapper
from helper.config import Configuration
from helper.data_loader import get_test_loader
from helper import utils

config = Configuration()

image_id = 1362

bnn = BNNWrapper()
bnn.load_model()

loss_fn = pyro.infer.Trace_ELBO(
    num_particles=config.bnn_adversary_samples
).differentiable_loss

nn = Network()
nn.load_model()

test_loader = get_test_loader(1, shuffle=False)
x, y = test_loader.dataset[image_id]

y = torch.tensor([y])

bnn_d, bnn_imgs, bnn_pertubation_imgs = bnn_adversary.run_attack(
    bnn, loss_fn, x, y, config.epsilons, image_id
)

nn_d, nn_imgs, nn_pertubation_imgs = nn_adversary.run_attack(
    nn, x, y, config.epsilons, 3
)

ids = [2, 5]

utils.img_show(x, f"Image, BNN Prediction: {bnn_d.iloc[0]['y_']}, NN Prediction: {nn_d.iloc[0]['y_']}")
for id in ids:
    utils.img_two_show(
        bnn_pertubation_imgs[id].cpu(),
        f"BNN Noise (epsilon: {bnn_d.iloc[id]['epsilon']})",
        nn_pertubation_imgs[id].cpu(),
        f"NN Noise (epsilon: {bnn_d.iloc[id]['epsilon']})",
    )
    utils.img_two_show(
        bnn_imgs[id].cpu(),
        f"Noise added to image, BNN Prediction: {bnn_d.iloc[id]['y_']}",
        nn_imgs[id].cpu(),
        f"Noise added to image, NN Prediction: {nn_d.iloc[id]['y_']}",
    )
