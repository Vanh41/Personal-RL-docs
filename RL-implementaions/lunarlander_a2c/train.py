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
num_episodes = 1000
actor = ActorNetwork(env)
actor_optimizer = optim.Adam(actor.parameters(), lr = learning_rate)
critic = CriticNetwork(env)
critic_optimizer = optim.Adam(critic.parameters(), lr = learning_rate)
memory = Memory()
maxsteps = 200

# train func
def train(memory, q_val):
    values = torch.stack(memory.values)
    q_vals = np.zeros(len(memory), 1)
    # target values are calculated backward
    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + (1-done)*gamma*q_val
        q_vals[len(memory)-1-i] = q_val # store values from the end to the beginning
    advantage = torch.Tensor(q_vals) - values
    #update actor
    actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    #update critic
    critic_loss = advantage.pow(2).mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
episode_reward = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action_probs = actor(torch.from_numpy(state))
        dist = torch.distributions.Categorical(probs = action_probs)
        log_prob = dist.sample()
        action = dist.sample()
        next_state, reward, done, info = env.step(action.detach().data.numpy())
        total_reward += reward
        steps += 1
        memory.add(log_prob, critic(state), reward, done)
        state = next_state
        if done or steps >= maxsteps:
            last_q_val = critic(torch.from_numpy(next_state))
            train(memory, last_q_val)
            memory.clear()
        episode_reward.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
            

torch.save(actor.state_dict(), 'actor_model.pth')
torch.save(critic.state_dict(), 'critic_model.pth')
        
    
    