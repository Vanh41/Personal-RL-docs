import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from memory import RolloutBuffer
from model import ActorNetwork, CriticNetwork

env = gym.make("LunarLander-v3")
# hyperparamters 
gamma = 0.99
learning_rate = 0.001
actor = ActorNetwork(env)
critic = CriticNetwork(env)
actor_optimizer = optim.Adam(actor.parameters(), lr = learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr = learning_rate)
critic = CriticNetwork(env)
buffer = RolloutBuffer()


