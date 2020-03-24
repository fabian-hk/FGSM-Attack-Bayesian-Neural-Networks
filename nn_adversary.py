from typing import List, Tuple
from pathlib import Path

import pandas

import torch
import torch.nn.functional as F

from networks import Network
from helper.data_loader import get_test_loader
from helper.adversary import fgsm_attack
from helper.config import Configuration


def run_attack(
    net: Network, x: torch.Tensor, y: torch.Tensor, epsilons: List[float], batch_id: int
) -> Tuple[pandas.DataFrame, List[torch.Tensor], List[torch.Tensor]]:
    x = x.to(net.device)
    y = y.to(net.device)

    x.requires_grad = True

    pred = net(x.view(-1, 28 * 28))

    loss = F.nll_loss(pred, y)
    net.zero_grad()
    loss.backward()
    data_grad = x.grad.data

    tmp_dict = {"id": [], "epsilon": [], "y": [], "y_": []}
    pertubed_images = []
    pertubation = []
    for epsilon in epsilons:
        pertubed_image, pert = fgsm_attack(x, epsilon, data_grad)
        pertubed_images.append(pertubed_image)
        pertubation.append(pert)

        pred = net(pertubed_image.view(-1, 28 * 28))

        y_ = pred.data.max(1).indices.item()

        tmp_dict["id"].append(batch_id)
        tmp_dict["epsilon"].append(epsilon)
        tmp_dict["y"].append(y.item())
        tmp_dict["y_"].append(y_)

    return pandas.DataFrame.from_dict(tmp_dict), pertubed_images, pertubation


if __name__ == "__main__":
    config = Configuration()

    net = Network()
    net.load_model()

    net.eval()

    test_loader = get_test_loader(batch_size=1, shuffle=False)

    result = []
    for batch_id, (x, y) in enumerate(test_loader):
        df, _ = run_attack(net, x, y, config.epsilons, batch_id)
        result.append(df)

        if batch_id % 100 == 0:
            print(f"Step {batch_id}/{len(test_loader.dataset)}")

    result_df = pandas.concat(result)  # type: pandas.DataFrame
    result_df.reset_index(inplace=True, drop=True)

    result_path = Path("data/")
    result_path.mkdir(exist_ok=True, parents=False)
    result_df.to_csv(result_path.joinpath(f"{config.id:02}_nn_result.csv"))
