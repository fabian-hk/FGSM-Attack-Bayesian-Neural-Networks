import torch
import pyro
import torch.nn.functional as F

from networks import BNNWrapper
from helper.data_loader import test_loader
from helper.adversary import fgsm_attack
from helper.utils import img_show

bnn = BNNWrapper()
bnn.load_model()

example, label = test_loader.dataset[10]
example_reshaped = example.unsqueeze(0)
example_reshaped.requires_grad = True

mean, var = bnn.predict(example_reshaped.view(-1, 28 * 28))
c = mean.max(1).indices.item()
print(f"Label: {label}, Prediction: {c}, Standard deviation: {var[0][c]}")


label = torch.tensor([label], dtype=torch.int64)

loss_fn = pyro.infer.Trace_ELBO(num_particles=20).differentiable_loss

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

pertubed_data = fgsm_attack(example_reshaped, 0.4, data_grad)
mean, var = bnn.predict(pertubed_data.view(-1, 28 * 28))

c = mean.max(1).indices.item()
print(f"Label: {label.item()}, Prediction: {c}, Standard deviation: {var [0][c]}")

img_show(pertubed_data)
