import torch
import pyro

from networks import BNNWrapper
from helper.data_loader import train_loader, test_loader

bnn = BNNWrapper()

optim = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(bnn.model, bnn.guide, optim, loss=pyro.infer.Trace_ELBO())

epochs = 5
for j in range(epochs):
    loss = 0
    for batch_id, (x, y) in enumerate(train_loader):
        x = x.to(bnn.device)
        y = y.to(bnn.device)
        # calculate the loss and take a gradient step
        loss += svi.step(x.view(-1, 28 * 28), y)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train

    print(f"Epoch: {j}, Loss: {total_epoch_loss_train}")

bnn.save_model()

correct = 0
total = 0
for j, (x, y) in enumerate(test_loader):
    x = x.to(bnn.device)
    y = y.to(bnn.device)
    mean, var = bnn.predict(x.view(-1, 28 * 28))
    total += y.size(0)
    correct += torch.eq(mean.max(1).indices, y).sum().item()

print(f"Accuracy: {correct/total}")
