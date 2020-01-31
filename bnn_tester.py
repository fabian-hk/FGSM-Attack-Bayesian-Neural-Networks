from pathlib import Path

import torch
import torch.nn.functional as F
import pyro

from bnn import guide, net, device, model
from data_loader import test_loader
from adversary import fgsm_attack

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
    mean = torch.mean(torch.stack(preds), 0)
    return mean


correct = 0
total = 0
for j, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    pred = predict(x.view(-1, 28 * 28))
    total += y.size(0)
    correct += torch.eq(pred.max(1).indices, y).sum().item()

print(f"Accuracy: {correct/total}")

example, label = test_loader.dataset[10]
example_reshaped = example.unsqueeze(0)
example_reshaped.requires_grad = True

pred = predict(example_reshaped.view(-1, 28*28).to(device))
print(f"Label: {label}, Prediction: {pred.max(1).indices.item()}")

label = torch.tensor(label).unsqueeze(0).to(device)
loss = F.nll_loss(pred, label)
net.zero_grad()
loss.backward()
data_grad = example_reshaped.grad.data

pertubed_data = fgsm_attack(example_reshaped, 0.5, data_grad)
pred = predict(pertubed_data.view(-1, 28*28).to(device))

print(f"Label: {label}, Prediction: {pred.item()}")
