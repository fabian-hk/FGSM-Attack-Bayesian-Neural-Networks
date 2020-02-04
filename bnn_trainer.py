import torch
import pyro

from networks import BNNWrapper
from helper.data_loader import get_train_loader, get_test_loader
from helper.config import Configuration


def train(
    svi: pyro.infer.SVI,
    bnn: BNNWrapper,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
):
    for batch_id, (x, y) in enumerate(train_loader):
        x = x.to(bnn.device)
        y = y.to(bnn.device)
        # calculate the loss and take a gradient step
        loss = svi.step(x.view(-1, 28 * 28), y)

        if batch_id % 100 == 0:
            print(
                f"Train Epoch: {epoch}, Step: {batch_id*len(x)}/{len(train_loader.dataset)}, Loss: {loss/x.size(0)}"
            )


def test(
    bnn: BNNWrapper,
    loss_fn: pyro.infer.Trace_ELBO,
    test_loader: torch.utils.data.DataLoader,
):
    correct = 0.0
    total = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for j, (x, y) in enumerate(test_loader):
            x = x.to(bnn.device)
            y = y.to(bnn.device)
            mean, var = bnn.predict(x.view(-1, 28 * 28))
            total += y.size(0)
            correct += torch.eq(mean.max(1).indices, y).sum().item()
            test_loss += loss_fn(
                bnn.model, bnn.guide, x_data=x.view(-1, 28 * 28), y_data=y
            )

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set, Average loss: {test_loss}, Accuracy: {float(correct) / float(len(test_loader.dataset))}\n"
    )


def training():
    config = Configuration()

    bnn = BNNWrapper()

    optim = pyro.optim.Adam({"lr": 0.01})
    svi = pyro.infer.SVI(bnn.model, bnn.guide, optim, loss=pyro.infer.Trace_ELBO())

    loss_fn = pyro.infer.Trace_ELBO(num_particles=20).differentiable_loss

    train_loader = get_train_loader()
    test_loader = get_test_loader()

    for epoch in range(config.bnn_training_epochs):
        train(svi, bnn, train_loader, epoch)
        test(bnn, loss_fn, test_loader)

    bnn.save_model()


if __name__ == "__main__":
    training()
