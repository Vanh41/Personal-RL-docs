import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(env.observation_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
            nn.Softmax(dim = -1)
        )
        self.cri
        