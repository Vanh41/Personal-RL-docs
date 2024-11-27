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
clip_epsilon = 0.2
actor = ActorNetwork(env)
critic = CriticNetwork(env)
actor_optimizer = optim.Adam(actor.parameters(), lr = learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr = learning_rate)
critic = CriticNetwork(env)
buffer = RolloutBuffer()


def collectTrajectories():
    state, _ = env.reset()
    done = False
    for i in range(10000):
        state = torch.FloatTensor(state)
        action_probs = actor(state)
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        next_state, reward, done, _, _ = env.step(action.item())
        value = critic(state)
        buffer.add(action, state, dist.log_prob(action), value, reward, done)
        state = next_state
        if done:
            break

def computeRTG():
    RTGs = []
    discounted_reward = 0
    for reward in reversed(buffer.rewards):
        discounted_reward = reward + gamma * discounted_reward
        RTGs.insert(0, discounted_reward)
    return torch.FloatTensor(RTGs)

def ppo_clip_loss(old_log_probs, new_log_probs, advantages):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss

def finish_episode():
    saved_actions = buffer.actions
    policy_losses = []
    value_losses = []
    returns = computeRTG()
    for (log_prob, value, reward, done) in zip(buffer.log_probs, buffer.values, returns, buffer.dones):
        advantage = reward - value.item()
        policy_losses.append()
    
    

        
        
