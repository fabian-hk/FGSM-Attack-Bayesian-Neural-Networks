from pathlib import Path

import pandas
import matplotlib.pyplot as plt
import numpy as np

import pyro
import torch

from helper.config import Configuration
from networks import Network, BNNWrapper
from helper.data_loader import get_test_loader
import nn_adversary
import bnn_adversary


def load_dict(name: str) -> pandas.DataFrame:
    path = Path(f"data/{name}")
    if not path.exists():
        raise FileExistsError()

    return pandas.read_csv(path)


def accuracy_over_epsilon(
    bnn_df: pandas.DataFrame, nn_df: pandas.DataFrame, config: Configuration
):
    bnn_y = []
    nn_y = []
    epsilons = config.epsilons
    for epsilon in epsilons:
        eps_df = bnn_df.loc[bnn_df["epsilon"] == epsilon]
        cor_df = eps_df.loc[eps_df["y"] == eps_df["y_"]]
        bnn_y.append(cor_df.shape[0] / eps_df.shape[0])

        eps_df = nn_df.loc[nn_df["epsilon"] == epsilon]
        cor_df = eps_df.loc[eps_df["y"] == eps_df["y_"]]
        nn_y.append(cor_df.shape[0] / eps_df.shape[0])

    plt.plot(epsilons, bnn_y, label="BNN")
    plt.plot(epsilons, nn_y, label="NN")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()

    plot_path = Path(f"data/{config.id:02}_accuracy.svg")
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def std_over_epsilon(bnn_df: pandas.DataFrame, config: Configuration):
    corr_y = []
    wro_y = []
    eps_y = []
    epsilons = config.epsilons
    for epsilon in epsilons:
        corr_eps = bnn_df.loc[
            (bnn_df["epsilon"] == epsilon) & (bnn_df["y"] == bnn_df["y_"])
        ]  # type: pandas.DataFrame
        wro_eps = bnn_df.loc[
            (bnn_df["epsilon"] == epsilon) & (bnn_df["y"] != bnn_df["y_"])
        ]  # type: pandas.DataFrame
        eps = bnn_df.loc[bnn_df["epsilon"] == epsilon]  # type: pandas.DataFrame
        corr_y.append(corr_eps["std"].mean())
        wro_y.append(wro_eps["std"].mean())
        eps_y.append(eps["std"].mean())

    config.correct_std = float(np.mean(corr_y))
    config.wrong_std = float(np.mean(wro_y))

    plt.plot(epsilons, corr_y, label="Average correct STD")
    plt.plot(epsilons, wro_y, label="Average wrong STD")
    plt.plot(epsilons, eps_y, label="Average STD")
    plt.xlabel("Epsilon")
    plt.ylabel("Standard Deviation")
    plt.legend()

    plot_path = Path(f"data/{config.id:02}_std.svg")
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def accuracy_over_epsilon_with_rejection(
    nn_df: pandas.DataFrame, bnn_df: pandas.DataFrame, config: Configuration
):
    bnn_y = []
    num_classified = []
    nn_y = []
    std_df = bnn_df.loc[(bnn_df["epsilon"] == 0.0) & (bnn_df["y"] == bnn_df["y_"])][
        "std"
    ]  # type: pandas.Series
    threshold = float(std_df.mean() + std_df.std())
    config.threshold_std = threshold
    epsilons = config.epsilons
    for epsilon in epsilons:
        eps_df = bnn_df.loc[bnn_df["epsilon"] == epsilon]
        classified_df = eps_df.loc[eps_df["std"] < threshold]
        cor_df = classified_df.loc[classified_df["y"] == classified_df["y_"]]
        bnn_y.append(cor_df.shape[0] / classified_df.shape[0])
        num_classified.append(classified_df.shape[0] / eps_df.shape[0])

        eps_df = nn_df.loc[nn_df["epsilon"] == epsilon]
        cor_df = eps_df.loc[eps_df["y"] == eps_df["y_"]]
        nn_y.append(cor_df.shape[0] / eps_df.shape[0])

    plt.plot(epsilons, bnn_y, label="BNN")
    plt.plot(epsilons, nn_y, label="NN")
    plt.plot(epsilons, num_classified, label="Percentage of classified examples")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()

    plot_path = Path(f"data/{config.id:02}_accuracy_with_rejection.svg")
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def plot_images(config: Configuration):
    bnn = BNNWrapper()
    bnn.load_model()

    loss_fn = pyro.infer.Trace_ELBO(
        num_particles=config.bnn_adversary_samples
    ).differentiable_loss

    nn = Network()
    nn.load_model()

    # sample images from the MNIST test data set
    imgs = [3, 4, 1362, 6930]

    fig, axes = plt.subplots(12, 2 * len(imgs), figsize=(36, 26))

    for j, img in enumerate(imgs):
        test_loader = get_test_loader(1, shuffle=False)
        x, y = test_loader.dataset[img]
        y = torch.tensor([y])

        bnn_d, bnn_imgs, _ = bnn_adversary.run_attack(
            bnn, loss_fn, x, y, config.epsilons, img
        )

        # font sizes
        title = 22
        body = 20

        index = 2 * j
        axes[0][index].set_title("Bayesian Neural Network", fontsize=title)
        for i in range(12):
            axes[i][index].imshow(
                bnn_imgs[i][0].cpu().detach(), cmap="gray", vmin=0, vmax=1
            )
            axes[i][index].set_xlabel(
                f"Label: {bnn_d['y'][i]}, Prediction: {bnn_d['y_'][i]}", fontsize=body
            )
            axes[i][index].set_ylabel(f"Eps: {bnn_d['epsilon'][i]}", fontsize=body)
            axes[i][index].set_yticklabels([])
            axes[i][index].set_xticklabels([])

        nn_d, nn_imgs, _ = nn_adversary.run_attack(nn, x, y, config.epsilons, 3)

        index = 2 * j + 1
        axes[0][index].set_title("Deep Neural Network", fontsize=title)
        for i in range(12):
            axes[i][index].imshow(
                nn_imgs[i][0].cpu().detach(), cmap="gray", vmin=0, vmax=1
            )
            axes[i][index].set_xlabel(
                f"Label: {nn_d['y'][i]}, Prediction: {nn_d['y_'][i]}", fontsize=body
            )
            axes[i][index].set_ylabel(f"Eps: {bnn_d['epsilon'][i]}", fontsize=body)
            axes[i][index].set_yticklabels([])
            axes[i][index].set_xticklabels([])

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)

    plot_path = Path(f"data/{config.id:02}_example_imgs.svg")
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def visualize():
    config = Configuration()

    bnn_df = load_dict(f"{config.id:02}_bnn_result.csv")
    nn_df = load_dict(f"{config.id:02}_nn_result.csv")

    accuracy_over_epsilon(bnn_df, nn_df, config)

    std_over_epsilon(bnn_df, config)

    accuracy_over_epsilon_with_rejection(nn_df, bnn_df, config)

    plot_images(config)

    config.save()


if __name__ == "__main__":
    visualize()
