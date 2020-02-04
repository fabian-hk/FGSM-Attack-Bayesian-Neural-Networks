import torch
import torch.nn.functional as F
import torch.optim as optim

from helper.data_loader import get_test_loader, get_train_loader
from networks import Network
from helper.config import Configuration


def train(net: Network, optimizer: torch.optim.Adam, train_loader: torch.utils.data.DataLoader, epoch: int):
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


def test(net: Network, test_loader: torch.utils.data.DataLoader):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(net.device)
            y = y.to(net.device)
            pred = net(x.view(-1, 28 * 28))
            test_loss += F.nll_loss(
                pred, y, size_average=False
            ).item()
            correct += torch.eq(pred.max(1).indices, y).sum().item()

        test_loss /= len(test_loader.dataset)
        print(
            f"\nTest set, Average loss: {test_loss}, Accuracy: {float(correct) / float(len(test_loader.dataset))}\n"
        )


def training():
    config = Configuration()

    net = Network()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    train_loader = get_train_loader()
    test_loader = get_test_loader()

    for epoch in range(config.nn_training_epochs):
        train(net, optimizer, train_loader, epoch)
        test(net, test_loader)

    net.save_model()


if __name__ == "__main__":
    training()
