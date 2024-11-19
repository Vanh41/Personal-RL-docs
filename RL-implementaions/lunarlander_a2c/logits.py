import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class Memory():
    def __init__(self):
        self.log_probs = []
        self.rewards = []
    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    def clear(self):
        self.log_probs.clear()
        self.rewards.clear()
    def _zip(self):
        return zip(self.log_probs, self.rewards)
    def __iter__(self):
        for data in self._zip():
            return data
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    def __len__(self):
        return len(self.rewards)