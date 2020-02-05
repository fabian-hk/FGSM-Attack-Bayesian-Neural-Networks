from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.distributions import Normal, Categorical

from helper.config import Configuration


class Network(nn.Module):
    def __init__(self, device: Optional[str] = "cpu"):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.out = nn.Linear(1024, 10)

        if torch.cuda.is_available():
            device = "cuda:0"

        self.to(device)
        self.device = device

        self.config = Configuration()

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return F.log_softmax(output, dim=1)

    def save_model(self):
        save_path = Path("data/saved_models/")
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), save_path.joinpath(f"{self.config.id:02}_model.pt"))

    def load_model(self):
        save_path = Path("data/saved_models/")
        if not save_path.exists():
            raise FileExistsError()

        self.load_state_dict(torch.load(save_path.joinpath(f"{self.config.id:02}_model.pt")))


class BNNWrapper(Network):
    def __init__(self, device: Optional[str] = "cpu"):
        super().__init__(device)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softplus = torch.nn.Softplus()

    def predict(
        self, x: torch.Tensor, num_samples: Optional[int] = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        sampled_models = [self.guide(None, None) for _ in range(num_samples)]
        preds = [model(x).data for model in sampled_models]
        stacked = torch.stack(preds)
        mean = torch.mean(stacked, 0)
        var = torch.std(stacked, 0)
        return mean, var

    def model(self, x_data: torch.Tensor, y_data: torch.Tensor):
        fc1w_prior = Normal(
            loc=torch.zeros_like(self.fc1.weight),
            scale=torch.ones_like(self.fc1.weight),
        )
        fc1b_prior = Normal(
            loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias)
        )

        outw_prior = Normal(
            loc=torch.zeros_like(self.out.weight),
            scale=torch.ones_like(self.out.weight),
        )
        outb_prior = Normal(
            loc=torch.zeros_like(self.out.bias), scale=torch.ones_like(self.out.bias)
        )

        priors = {
            "fc1.weight": fc1w_prior,
            "fc1.bias": fc1b_prior,
            "out.weight": outw_prior,
            "out.bias": outb_prior,
        }

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()

        lhat = self.log_softmax(lifted_reg_model(x_data))

        pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

    def guide(self, x_data: torch.Tensor, y_data: torch.Tensor):
        # First layer weight distribution priors
        fc1w_mu = torch.randn_like(self.fc1.weight)
        fc1w_sigma = torch.randn_like(self.fc1.weight)
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = self.softplus(pyro.param("fc1w_sigma", fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        # First layer bias distribution priors
        fc1b_mu = torch.randn_like(self.fc1.bias)
        fc1b_sigma = torch.randn_like(self.fc1.bias)
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = self.softplus(pyro.param("fc1b_sigma", fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
        # Output layer weight distribution priors
        outw_mu = torch.randn_like(self.out.weight)
        outw_sigma = torch.randn_like(self.out.weight)
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = self.softplus(pyro.param("outw_sigma", outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
        # Output layer bias distribution priors
        outb_mu = torch.randn_like(self.out.bias)
        outb_sigma = torch.randn_like(self.out.bias)
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = self.softplus(pyro.param("outb_sigma", outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
        priors = {
            "fc1.weight": fc1w_prior,
            "fc1.bias": fc1b_prior,
            "out.weight": outw_prior,
            "out.bias": outb_prior,
        }

        lifted_module = pyro.random_module("module", self, priors)

        return lifted_module()

    def save_model(self):
        # save parameters from the pyro module not pytorch itself
        save_path = Path("data/saved_models/")
        save_path.mkdir(exist_ok=True, parents=True)
        pyro.get_param_store().save(save_path.joinpath(f"{self.config.id:02}_bnn_params.pr"))

    def load_model(self):
        save_path = Path("data/saved_models/")
        if not save_path.exists():
            raise FileExistsError()

        pyro.get_param_store().load(save_path.joinpath(f"{self.config.id:02}_bnn_params.pr"))
        pyro.module("module", self, update_module_params=True)


class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()
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
