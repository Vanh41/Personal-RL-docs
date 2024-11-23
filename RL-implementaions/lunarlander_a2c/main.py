import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from collections import namedtuple
from memory import RolloutBuffer
from model import ActorNetwork, CriticNetwork

env = gym.make("LunarLander-v3", render_mode="human")

# Load the trained models
actor = ActorNetwork(env)
critic = CriticNetwork(env)
actor.load_state_dict(torch.load('/Users/vietanh/Documents/Personal-RL-docs/RL-implementaions/lunarlander_a2c/actor_model.pth'))
critic.load_state_dict(torch.load('/Users/vietanh/Documents/Personal-RL-docs/RL-implementaions/lunarlander_a2c/critic_model.pth'))

def run_episode(env, actor):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state)  # Convert state to torch.Tensor
        action_probs = actor(state)
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample().item()  # Sample an action from the distribution
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    return total_reward

if __name__ == '__main__':
    num_episodes = 1
    for i in range(num_episodes):
        total_reward = run_episode(env, actor)
        print(f"Episode {i + 1}: Total Reward: {total_reward}")