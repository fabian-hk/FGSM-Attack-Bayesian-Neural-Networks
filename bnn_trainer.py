import torch
import pyro

from networks import BNNWrapper
from helper.data_loader import train_loader, test_loader


def train(svi: pyro.infer.SVI, bnn: BNNWrapper, epoch: int):
    for batch_id, (x, y) in enumerate(train_loader):
        x = x.to(bnn.device)
        y = y.to(bnn.device)
        # calculate the loss and take a gradient step
        loss = svi.step(x.view(-1, 28 * 28), y)

        if batch_id % 100 == 0:
            print(
                f"Train Epoch: {epoch}, Step: {batch_id*len(x)}/{len(train_loader.dataset)}, Loss: {loss/x.size(0)}"
            )


def test(bnn: BNNWrapper):
    correct = 0.0
    total = 0.0
    for j, (x, y) in enumerate(test_loader):
        x = x.to(bnn.device)
        y = y.to(bnn.device)
        mean, var = bnn.predict(x.view(-1, 28 * 28))
        total += y.size(0)
        correct += torch.eq(mean.max(1).indices, y).sum().item()

    print(f"Test set accuracy: {correct / total}")


def training():
    bnn = BNNWrapper()

    optim = pyro.optim.Adam({"lr": 0.01})
    svi = pyro.infer.SVI(bnn.model, bnn.guide, optim, loss=pyro.infer.Trace_ELBO())

    epochs = 5
    for epoch in range(epochs):
        train(svi, bnn, epoch)
        test(bnn)

    bnn.save_model()


if __name__ == "__main__":
    training()
