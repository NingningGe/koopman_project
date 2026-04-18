import os
import random
import time
# from dataclasses import dataclass

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
#6/20
N=20

if N==7:
    class QNetwork(nn.Module):
        def __init__(self, env):
            super().__init__()
            self.fc1 = nn.Linear(21, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x, a):
            x = torch.cat([x, a], 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    high = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]).astype(np.float32).reshape(1,7)
    low = np.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4]).astype(np.float32).reshape(1,7)
    class Actor(nn.Module):
        def __init__(self, env):
            super().__init__()
            self.fc1 = nn.Linear(14, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_mu = nn.Linear(256, 7)
            # action rescaling
            self.register_buffer(
                "action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32)
            )

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc_mu(x))
            return x * self.action_scale + self.action_bias

elif N==6:
    class QNetwork(nn.Module):
        def __init__(self, env):
            super().__init__()
            self.fc1 = nn.Linear(18, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x, a):
            x = torch.cat([x, a], 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    high = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]).astype(np.float32).reshape(1,6)
    low = np.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4]).astype(np.float32).reshape(1,6)
    class Actor(nn.Module):
        def __init__(self, env):
            super().__init__()
            self.fc1 = nn.Linear(12, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_mu = nn.Linear(256, 6)
            # action rescaling
            self.register_buffer(
                "action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32)
            )

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc_mu(x))
            return x * self.action_scale + self.action_bias
        
elif N==20:
    class QNetwork(nn.Module):
        def __init__(self, env):
            super().__init__()
            self.fc1 = nn.Linear(30, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x, a):
            x = torch.cat([x, a], 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    high = np.array([0.4189, 0.4068, 0.0370, 0.3453, 0.3644, 0.4608, 0.1827, 0.1348, 0.4151, 0.4022]).astype(np.float32).reshape(1,10)
    low = np.array([-0.4533, -0.4103, -0.0397, -0.4026, -0.3591, -0.3698, -0.1891, -0.1338, -0.4911, -0.3657]).astype(np.float32).reshape(1,10)
    class Actor(nn.Module):
        def __init__(self, env):
            super().__init__()
            self.fc1 = nn.Linear(20, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_mu = nn.Linear(256, 10)
            # action rescaling
            self.register_buffer(
                "action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32)
            )

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc_mu(x))
            return x * self.action_scale + self.action_bias
