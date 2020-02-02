from pathlib import Path
import pickle

import matplotlib.pyplot as plt


def load_dict(name: str):
    path = Path(f"data/{name}")
    if not path.exists():
        raise FileExistsError()

    return pickle.load(path.open("rb"))


def visualize():
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    bnn_dict = load_dict("bnn_result.p")
    nn_dict = load_dict("nn_result.p")

    bnn_y = []
    nn_y = []
    for epsilon in epsilons:
        correct = float(bnn_dict[f"{epsilon}_correct"])
        wrong = float(bnn_dict[f"{epsilon}_wrong"])
        bnn_y.append(correct / (correct + wrong))

        correct = float(nn_dict[f"{epsilon}_correct"])
        wrong = float(nn_dict[f"{epsilon}_wrong"])
        nn_y.append(correct / (correct + wrong))

    plt.plot(epsilons, bnn_y)
    plt.plot(epsilons, nn_y)
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")

    plot_path = Path("data/accuracy.svg")
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    visualize()
