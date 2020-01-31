import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pyro
from pyro.distributions import Normal, Categorical

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output


net = NN(28 * 28, 1024, 10)
net.to(device)

log_softmax = nn.LogSoftmax(dim=1)


def model(x_data, y_data):
    fc1w_prior = Normal(
        loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight)
    )
    fc1b_prior = Normal(
        loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias)
    )

    outw_prior = Normal(
        loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight)
    )
    outb_prior = Normal(
        loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias)
    )

    priors = {
        "fc1.weight": fc1w_prior,
        "fc1.bias": fc1b_prior,
        "out.weight": outw_prior,
        "out.bias": outb_prior,
    }

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = log_softmax(lifted_reg_model(x_data))

    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)


softplus = torch.nn.Softplus()


def guide(x_data, y_data):
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {
        "fc1.weight": fc1w_prior,
        "fc1.bias": fc1b_prior,
        "out.weight": outw_prior,
        "out.bias": outb_prior,
    }

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()
