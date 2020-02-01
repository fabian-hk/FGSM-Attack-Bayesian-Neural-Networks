from pathlib import Path

import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cnn_trainer import Net
from data_loader import test_loader
from adversary import fgsm_attack

batch_size = 128


model = Net()

save_path = Path("saved_models/")
if not save_path.exists():
    raise FileExistsError()
model.load_state_dict(torch.load(save_path.joinpath("model.pt")))
model.eval()

example, label = test_loader.dataset[10]
example_reshaped = example.unsqueeze(0)
example_reshaped.requires_grad = True

pred = model(example_reshaped.view(-1, 28 * 28))
print(pred.data.max(1, keepdim=True))

label = torch.tensor(label).unsqueeze(0)
loss = F.nll_loss(pred, label)
model.zero_grad()
loss.backward()
data_grad = example_reshaped.grad.data

pertubed_data = fgsm_attack(example_reshaped, 0.09, data_grad)

pred = model(pertubed_data.view(-1, 28 * 28))
print(pred.data.max(1, keepdim=True))


