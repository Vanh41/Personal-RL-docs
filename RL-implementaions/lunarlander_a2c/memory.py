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
        self.values = []
        self.rewards = []
        self.dones = []
    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
    def _zip(self):
        return zip(self.log_probs, self.values, self.rewards, self.dones)
    def __iter__(self):
        for data in self._zip():
            return data
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    def __len__(self):
        return len(self.rewards)
    

class RolloutBuffer():
    def __init__(self):
        self.saved_action = []
        self.rewards = []
    def add(self, saved_action, reward):
        self.saved_action.append(saved_action)
        self.rewards.append(reward)
    def clear(self):
        self.saved_action.clear()
        self.rewards.clear()
    def _zip(self):
        return zip(self.saved_action, self.rewards)
    def __iter__(self):
        for data in self.saved_action:
            return data
    def __len__(self):
        return len(self.rewards)