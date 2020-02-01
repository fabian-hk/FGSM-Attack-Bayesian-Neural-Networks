import torch
import torch.nn.functional as F
import torch.optim as optim

from helper.data_loader import test_loader, train_loader
from networks import Network


def train(net: Network, optimizer: torch.optim.Adam, epoch: int):
    net.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(x.view(-1, 28 * 28).to(net.device))
        loss = F.nll_loss(output, y.to(net.device))
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch}, Step: {batch_idx*len(x)}/{len(train_loader.dataset)}, Loss: {loss.item()}"
            )


def test(net: Network):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data.view(-1, 28 * 28).to(net.device))
            test_loss += F.nll_loss(
                output, target.to(net.device), size_average=False
            ).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(net.device).data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        print(
            f"Test set: Avg. loss: {test_loss}, Accuracy: {float(correct)/float(len(test_loader.dataset))}"
        )


def training():
    n_epochs = 2

    net = Network()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        train(net, optimizer, epoch)
        test(net)

    net.save_model()


if __name__ == "__main__":
    training()
