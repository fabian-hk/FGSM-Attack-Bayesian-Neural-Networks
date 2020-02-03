from pathlib import Path

import pandas
import matplotlib.pyplot as plt
import numpy as np

from helper.config import Configuration


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

    plot_path = Path(f"data/{config.id}_accuracy.svg")
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def std_over_epsilon(bnn_df: pandas.DataFrame, config: Configuration):
    corr_y = []
    wro_y = []
    epsilons = config.epsilons
    for epsilon in epsilons:
        corr_eps = bnn_df.loc[
            (bnn_df["epsilon"] == epsilon) & bnn_df["y"] == bnn_df["y_"]
        ]  # type: pandas.DataFrame
        wro_eps = bnn_df.loc[
            (bnn_df["epsilon"] == epsilon) & bnn_df["y"] != bnn_df["y_"]
        ]  # type: pandas.DataFrame
        corr_y.append(corr_eps["std"].mean())
        wro_y.append(wro_eps["std"].mean())

    config.correct_std = float(np.mean(corr_y))
    config.wrong_std = float(np.mean(wro_y))

    plt.plot(epsilons, corr_y, label="Correct STD")
    plt.plot(epsilons, wro_y, label="Wrong STD")
    plt.xlabel("Epsilon")
    plt.ylabel("STD")
    plt.legend()

    plot_path = Path(f"data/{config.id}_std.svg")
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

    plot_path = Path(f"data/{config.id}_accuracy_with_rejection.svg")
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def visualize():
    config = Configuration()

    bnn_df = load_dict("bnn_result.csv")
    nn_df = load_dict("nn_result.csv")

    accuracy_over_epsilon(bnn_df, nn_df, config)

    std_over_epsilon(bnn_df, config)

    accuracy_over_epsilon_with_rejection(nn_df, bnn_df, config)

    config.save()


if __name__ == "__main__":
    visualize()
