from pathlib import Path

import torch
import pyro

from bnn import guide, net, device, model
from data_loader import test_loader
from adversary import fgsm_attack
from utils import img_show

save_path = Path("saved_models/")
if not save_path.exists():
    raise FileExistsError()

pyro.get_param_store().load(save_path.joinpath("bnn_params.pr"))
pyro.module("module", net, update_module_params=True)

net.to(device)


def predict(x):
    num_samples = 10
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    preds = [model(x).data for model in sampled_models]
    stacked = torch.stack(preds)
    mean = torch.mean(stacked, 0)
    var = torch.std(stacked, 0)
    return mean, var


"""
correct = 0
total = 0
for j, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    pred, _ = predict(x.view(-1, 28 * 28))
    total += y.size(0)
    correct += torch.eq(pred.max(1).indices, y).sum().item()

print(f"Accuracy: {correct/total}")
"""

example, label = test_loader.dataset[10]
example_reshaped = example.unsqueeze(0)
example_reshaped.requires_grad = True

mean, var = predict(example_reshaped.view(-1, 28 * 28).to(device))
c = mean.max(1).indices.item()
print(f"Label: {label}, Prediction: {c}, Variance: {var[0][c]}")

optim = pyro.optim.Adam({"lr": 0.0})
svi = pyro.infer.SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO())

label = torch.Tensor([label])
svi.step(x_data=example_reshaped.view(-1, 28 * 28).to(device), y_data=label.to(device))

data_grad = example_reshaped.grad.data

pertubed_data = fgsm_attack(example_reshaped, 0.5, data_grad)
mean, var = predict(pertubed_data.view(-1, 28 * 28).to(device))

c = mean.max(1).indices.item()
print(f"Label: {label.item()}, Prediction: {c}, Variance: {var [0][c]}")

img_show(pertubed_data)
