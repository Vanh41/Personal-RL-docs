import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class RolloutBuffer():
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    def add(self, action, state, log_prob, value, reward, done):
        self.actions.append(action)
        self.actions.append(state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()