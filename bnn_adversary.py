from typing import List, Tuple
from pathlib import Path

import pandas

import torch
import pyro

from networks import BNNWrapper
from helper.data_loader import get_test_loader
from helper.adversary import fgsm_attack
from helper.config import Configuration


def run_attack(
    bnn: BNNWrapper,
    loss_fn: pyro.infer.Trace_ELBO,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilons: List[float],
    batch_id: int,
) -> Tuple[pandas.DataFrame, List[torch.Tensor]]:
    x = x.to(bnn.device)
    y = y.to(bnn.device)

    x.requires_grad = True

    loss = loss_fn(bnn.model, bnn.guide, x_data=x.view(-1, 28 * 28), y_data=y)
    loss.backward()

    data_grad = x.grad.data

    tmp_dict = {"id": [], "epsilon": [], "y": [], "y_": [], "std": []}
    pertubed_images = []
    for epsilon in epsilons:
        pertubed_image = fgsm_attack(x, epsilon, data_grad)
        pertubed_images.append(pertubed_image)
        mean, std = bnn.predict(pertubed_image.view(-1, 28 * 28))

        y_ = mean.max(1).indices.item()
        std_ = std[0][y_].item()

        tmp_dict["id"].append(batch_id)
        tmp_dict["epsilon"].append(epsilon)
        tmp_dict["y"].append(y.item())
        tmp_dict["y_"].append(y_)
        tmp_dict["std"].append(std_)

    return pandas.DataFrame.from_dict(tmp_dict), pertubed_images


if __name__ == "__main__":
    config = Configuration()

    bnn = BNNWrapper()
    bnn.load_model()
    loss_fn = pyro.infer.Trace_ELBO(
        num_particles=config.bnn_adversary_samples
    ).differentiable_loss

    test_loader = get_test_loader(batch_size=1, shuffle=False)

    result = []  # type: List[pandas.DataFrame]
    for batch_id, (x, y) in enumerate(test_loader):
        result.append(run_attack(bnn, loss_fn, x, y, config.epsilons, batch_id))

        if batch_id % 100 == 0:
            print(f"Step {batch_id}/{len(test_loader.dataset)}")

    result_df = pandas.concat(result)  # type: pandas.DataFrame
    result_df.reset_index(inplace=True, drop=True)

    result_path = Path("data/")
    result_path.mkdir(exist_ok=True, parents=False)
    result_df.to_csv(result_path.joinpath(f"{config.id:02}_bnn_result.csv"))
