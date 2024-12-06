import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from memory import RolloutBuffer
from model import ActorNetwork, CriticNetwork
from minigrid.wrappers import FlatObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper


env = gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode = 'rgb_array')

n_frames = 10**6
n_epochs = 4
gamma = 0.99
learning_rate = 0.001
value_loss_coef = 0.5
entropy_coef = 0.01
epsilon = 1e-8
gae_lambda = 0.95
clip_epsilon = 0.2
batch_size = 10
num_agents = 16
num_runs = 30

actor = ActorNetwork(env)
critic = CriticNetwork(env)
actor_optimizer = optim.Adam(actor.parameters(), lr = learning_rate, eps = epsilon)
critic_optimizer = optim.Adam(critic.parameters(), lr = learning_rate, eps = epsilon)
buffer = RolloutBuffer(batch_size)


def chooseAction(state):
    state = torch.FloatTensor(state)
    action_probs = actor(state)


