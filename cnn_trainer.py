from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from data_loader import test_loader, train_loader

n_epochs = 2
batch_size = 128

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


class Net_CNN(nn.Module):
    def __init__(self):
        super(Net_CNN, self).__init__()
        self.con1 = nn.Conv2d(1, 10, kernel_size=5)
        self.con2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, input, **kwargs):
        x = F.relu(F.max_pool2d(self.con1(input), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.con2(x)), 2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.out = nn.Linear(1024, 10)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return F.log_softmax(output, dim=1)


network = Net()
network.to(device)
optimizer = optim.Adam(network.parameters(), lr=0.01)


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data.view(-1, 28 * 28).to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch}, Step: {batch_idx*len(data)}/{len(train_loader.dataset)}, Loss: {loss.item()}"
            )


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.view(-1, 28 * 28).to(device))
            test_loss += F.nll_loss(output, target.to(device), size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        print(
            f"Test set: Avg. loss: {test_loss}, Accuracy: {float(correct)/float(len(test_loader.dataset))}"
        )


if __name__ == "__main__":
    for epoch in range(n_epochs):
        train(epoch)
        test()

    save_path = Path("saved_models/")
    save_path.mkdir(exist_ok=True, parents=False)
    torch.save(network.state_dict(), save_path.joinpath("model.pt"))
