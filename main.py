import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import torch_utils


class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        return self.map3(x)


class Discriminator(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        return F.sigmoid(self.map3(x))


def get_real_input_data(size):
    """
    Arguments:
        size(int): Number of instances to generate

    Returs:
        returns a tensor of size [size * N], where N is the
        length of a single instances' vector
    """
    return torch_utils.from_numpy(np.random.normal(size=(size, 1))).float()


def main():
    g_input_size = 1  # some source of noise
    g_hidden_size = 100
    g_output_size = 1

    d_input_size = 100
    d_hidden_size = 100

    num_epoch = 1000
    d_steps = 1
    g_steps = 1
    criterion = nn.BCELoss()

    G = Generator(
            input_size=g_input_size,
            hidden_size=g_hidden_size,
            output_size=g_output_size)
    D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size)

    g_optimizer = optim.Adam(G.parameters())
    d_optimizer = optim.Adam(D.parameters())

    for epoch in range(10000):

        for d_index in range(d_steps):
            D.zero_grad()

            # 1A. Training on real data
            d_real_data = autograd.Variable(get_real_input_data(d_input_size))
            d_real_decision = D(d_real_data.t())
            d_real_error = criterion(d_real_decision, autograd.Variable(torch.ones(1, 1)))
            d_real_error.backward()

            # 1B. Training on fake data
            generator_seed = torch_utils.from_numpy(
                    np.random.rand(d_input_size, g_input_size)).float()
            d_fake_data = G(autograd.Variable(generator_seed)).detach()
            d_fake_decision = D(d_fake_data.t())
            d_fake_error = criterion(d_fake_decision, autograd.Variable(torch.zeros(1, 1)))
            d_fake_error.backward()

            d_optimizer.step()

        for g_index in range(g_steps):
            G.zero_grad()

            generator_seed = torch_utils.from_numpy(
                    np.random.rand(d_input_size, g_input_size)).float()
            g_fake_data = G(autograd.Variable(generator_seed))
            dg_fake_decision = D(g_fake_data.t())
            g_error = criterion(dg_fake_decision, autograd.Variable(torch.ones(1, 1)))

            g_error.backward()
            g_optimizer.step()

        print("Epoch = %d G_loss = %.3f D_loss (fake) = %.3f D_loss (real) = %.3f"
                % (epoch, g_error, d_fake_error, d_real_error))


if __name__ == "__main__":
    main()
