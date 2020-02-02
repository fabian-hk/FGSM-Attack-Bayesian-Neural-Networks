from pathlib import Path

import pandas
import matplotlib.pyplot as plt


def load_dict(name: str) -> pandas.DataFrame:
    path = Path(f"data/{name}")
    if not path.exists():
        raise FileExistsError()

    return pandas.read_csv(path)


def visualize():
    epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    bnn_df = load_dict("bnn_result.csv")
    nn_df = load_dict("nn_result.csv")

    bnn_y = []
    nn_y = []
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

    plot_path = Path("data/accuracy.svg")
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    visualize()
