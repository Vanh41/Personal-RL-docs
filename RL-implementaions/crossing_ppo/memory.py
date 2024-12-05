import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym

class RolloutBuffer():
    def __init__(self, batch_size):
        self.actions = []
        self.states = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)        
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.actions),\
                np.array(self.states),\
                np.array(self.probs),\
                np.array(self.values),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
    
    def store_memory(self, state, action, prob, value, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()