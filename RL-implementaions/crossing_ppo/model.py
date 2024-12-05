import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class ActorNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(env.observation_space.shape[0], 32, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (5, 5)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 11 * 11, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
            nn.Softmax(dim = -1)
        )
    def forward(self, x):
        return self.network(x)
    
class CriticNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(env.observation_space.shape[0], 32, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (5, 5)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 11 * 11, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.network(x)