from typing import List, Dict
from pathlib import Path
import pickle

import torch
import torch.nn.functional as F

from networks import Network
from helper.data_loader import get_test_loader
from helper.adversary import fgsm_attack
from helper.utils import img_show


def run_attack(net: Network, x: torch.Tensor, y: torch.Tensor, epsilons: List[float], result: Dict):
    x = x.to(net.device)
    y = y.to(net.device)

    x.requires_grad = True

    pred = net(x.view(-1, 28 * 28))
    print(f"No attack: Label = {y.item()}, Prediction: {pred.data.max(1).indices.item()}")

    loss = F.nll_loss(pred, y)
    net.zero_grad()
    loss.backward()
    data_grad = x.grad.data

    for epsilon in epsilons:
        pertubed_data = fgsm_attack(x, epsilon, data_grad)

        pred = net(pertubed_data.view(-1, 28 * 28))

        if pred.max(1).indices == y:
            result[f"{epsilon}_correct"] += 1
        else:
            result[f"{epsilon}_wrong"] += 1

        print(f"Model under attack: Label = {y.item()}, Prediction: {pred.data.max(1).indices.item()}")

        img_show(pertubed_data.cpu())


if __name__ == "__main__":
    net = Network()
    net.load_model()

    net.eval()

    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    result = {}
    for epsilon in epsilons:
        result[f"{epsilon}_correct"] = 0
        result[f"{epsilon}_wrong"] = 0

    for x, y in get_test_loader(1):
        run_attack(net, x, y, epsilons, result)
        break

    result_path = Path("data/")
    result_path.mkdir(exist_ok=True, parents=False)
    pickle.dump(result, result_path.joinpath("nn_result.p").open("wb"))

    print(result)
