from typing import List, Dict
from pathlib import Path
import pickle

import torch
import pyro

from networks import BNNWrapper
from helper.data_loader import get_test_loader
from helper.adversary import fgsm_attack


def run_attack(
    bnn: BNNWrapper,
    loss_fn: pyro.infer.Trace_ELBO,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilons: List[float],
    result: Dict,
):
    x = x.to(bnn.device)
    y = y.to(bnn.device)

    x.requires_grad = True

    loss = loss_fn(bnn.model, bnn.guide, x_data=x.view(-1, 28 * 28), y_data=y)
    loss.backward()

    data_grad = x.grad.data

    for epsilon in epsilons:
        pertubed_data = fgsm_attack(x, epsilon, data_grad)
        mean, std = bnn.predict(pertubed_data.view(-1, 28 * 28))

        c = mean.max(1).indices.item()

        if mean.max(1).indices == y:
            result[f"{epsilon}_correct"] += 1
        else:
            result[f"{epsilon}_wrong"] += 1

        result[f"{epsilon}_std"].append(std[0][c].item())

        print(f"Label: {y.item()}, Prediction: {c}, Standard deviation: {std[0][c]}")


if __name__ == "__main__":
    bnn = BNNWrapper()
    bnn.load_model()
    loss_fn = pyro.infer.Trace_ELBO(num_particles=20).differentiable_loss

    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    result = {}
    for epsilon in epsilons:
        result[f"{epsilon}_correct"] = 0
        result[f"{epsilon}_wrong"] = 0
        result[f"{epsilon}_std"] = []

    for x, y in get_test_loader(1):
        run_attack(bnn, loss_fn, x, y, epsilons, result)

    result_path = Path("data/")
    result_path.mkdir(exist_ok=True, parents=False)
    pickle.dump(result, result_path.joinpath("bnn_result.p").open("wb"))
