import torch
import pyro

from networks import BNNWrapper
from helper.data_loader import test_loader
from helper.adversary import fgsm_attack
from helper.utils import img_show

bnn = BNNWrapper()
bnn.load_model()

"""
correct = 0
total = 0
for j, (x, y) in enumerate(test_loader):
    x = x.to(bnn.device)
    y = y.to(bnn.device)
    mean, _ = bnn.predict(x.view(-1, 28 * 28))
    total += y.size(0)
    correct += torch.eq(mean.max(1).indices, y).sum().item()

print(f"Accuracy: {correct/total}")
"""

example, label = test_loader.dataset[10]
example_reshaped = example.unsqueeze(0)
example_reshaped.requires_grad = True

mean, var = bnn.predict(example_reshaped.view(-1, 28 * 28))
c = mean.max(1).indices.item()
print(f"Label: {label}, Prediction: {c}, Variance: {var[0][c]}")

loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

label = torch.Tensor([label])
loss = loss_fn(
    bnn.model,
    bnn.guide,
    x_data=example_reshaped.view(-1, 28 * 28).to(bnn.device),
    y_data=label.to(bnn.device),
)
print(f"Loss: {loss}")
loss.backward()

data_grad = example_reshaped.grad.data
print(f"Gradient: {data_grad.sum()}")

pertubed_data = fgsm_attack(example_reshaped, 0.3, data_grad)
mean, var = bnn.predict(pertubed_data.view(-1, 28 * 28))

c = mean.max(1).indices.item()
print(f"Label: {label.item()}, Prediction: {c}, Variance: {var [0][c]}")

img_show(pertubed_data)
