import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import copy
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("../utility/")
from scipy.integrate import odeint
import time

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# define network
def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class Network(nn.Module):
    def __init__(self, encode_layers, Nkoopman, u_dim):
        super(Network, self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
            if layer_i != len(encode_layers) - 2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim, Nkoopman, bias=False)

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], dim=-1)

    def forward(self, x, u):
        return self.lA(x) + self.lB(u)


class ENC_net(nn.Module):
    def __init__(self, ENC_layers):
        super(ENC_net, self).__init__()
        ENCLayers = OrderedDict()
        for layer_i in range(len(ENC_layers) - 1):
            ENCLayers["linear_{}".format(layer_i)] = nn.Linear(ENC_layers[layer_i], ENC_layers[layer_i + 1])
            if layer_i != len(ENC_layers) - 2:
                ENCLayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.ENC_net = nn.Sequential(ENCLayers)

    def ENC(self, x):
        return self.ENC_net(x)


class DEC_net(nn.Module):
    def __init__(self, DEC_layers):
        super(DEC_net, self).__init__()
        DECLayers = OrderedDict()
        for layer_i in range(len(DEC_layers) - 1):
            DECLayers["linear_{}".format(layer_i)] = nn.Linear(DEC_layers[layer_i], DEC_layers[layer_i + 1])
            if layer_i != len(DEC_layers) - 2:
                DECLayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.DEC_net = nn.Sequential(DECLayers)

    def DEC(self, x):
        return self.DEC_net(x)