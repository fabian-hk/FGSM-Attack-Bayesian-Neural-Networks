from pathlib import Path

import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from bnn import net, model, guide, device
from data_loader import train_loader, test_loader

optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

epochs = 5
loss = 0

for j in range(epochs):
    loss = 0
    for batch_id, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        # calculate the loss and take a gradient step
        loss += svi.step(x.view(-1, 28 * 28), y)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train

    print(f"Epoch: {j}, Loss: {total_epoch_loss_train}")

# save parameters from the pyro module not pytorch itself
save_path = Path("saved_models/")
save_path.mkdir(exist_ok=True, parents=False)
pyro.get_param_store().save(save_path.joinpath("bnn_params.pr"))


def predict(x):
    num_samples = 10
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    preds = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(preds), 0)
    return mean.max(1).indices


correct = 0
total = 0
for j, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    pred = predict(x.view(-1, 28 * 28))
    total += y.size(0)
    correct += torch.eq(pred, y).sum().item()

print(f"Accuracy: {correct/total}")
