import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
from buffers import ActorCritic
from model import RolloutBuffer

class PPOAgent:
     def __init__(self, env, alpha=0.0003, gamma=0.99, gae_lambda=0.95, policy_clip=0.2, batch_size=2000000, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.network = ActorCritic(env=env)
        self.buffer = RolloutBuffer(batch_size)
        