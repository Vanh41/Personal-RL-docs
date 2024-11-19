import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from memory import Memory
from model import ActorNetwork, CriticNetwork

env = gym.make("LunarLander-v3")

# hyperparameters

gamma = 0.99
learning_rate = 0.001
num_episodes = 100
actor = ActorNetwork(env)
actor_optimizer = optim.Adam(actor.parameters(), lr = learning_rate)
critic = CriticNetwork(env)
critic_optimizer = optim.Adam(critic.parameters(), lr = learning_rate)
memory = Memory()
maxsteps = 200

def selectAction(state):
    state = torch.from_numpy(state).float()
    action_probs = actor(state)
    state_value = critic(state)
    # create a categorical distribution over the list of probabilities of actions
    m = torch.distributions.Categorical(action_probs)
    # sample an act
    action = m.sample()
    # save to action buffer 

    
    



torch.save(actor.state_dict(), 'actor_model.pth')
torch.save(critic.state_dict(), 'critic_model.pth')
        
    
    