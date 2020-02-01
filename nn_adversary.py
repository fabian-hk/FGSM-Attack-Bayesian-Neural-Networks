import torch
import torch.nn.functional as F

from networks import Network
from helper.data_loader import test_loader
from helper.adversary import fgsm_attack
from helper.utils import img_show

batch_size = 128


net = Network()
net.load_model()

net.eval()

example, label = test_loader.dataset[10]
example_reshaped = example.unsqueeze(0)
example_reshaped.requires_grad = True

pred = net(example_reshaped.view(-1, 28 * 28).to(net.device))
print(f"No attack: Label = {label}, Prediction: {pred.data.max(1).indices.item()}")

label = torch.tensor(label).unsqueeze(0)
loss = F.nll_loss(pred.to(net.device), label.to(net.device))
net.zero_grad()
loss.backward()
data_grad = example_reshaped.grad.data

pertubed_data = fgsm_attack(example_reshaped, 0.3, data_grad)

pred = net(pertubed_data.view(-1, 28 * 28).to(net.device))
print(f"Model under attack: Label = {label.item()}, Prediction: {pred.data.max(1).indices.item()}")

img_show(pertubed_data)
