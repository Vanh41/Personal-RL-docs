import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class ActorNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
            nn.Softmax(dim = -1)
        )
    def forward(self, x):
        return self.network(x)
    
class CriticNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.network(x)
    
#hyperparameter
gamma = 0.99
learning_rate = 0.001
batch_size = 64
num_epochs = 1000

class A2C:
    def __init__(self, env):
        self.env = env
        self.actor = ActorNetwork(env)
        self.critic = CriticNetwork(env)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = learning_rate)
    def train(self, num_epochs, batch_size):
        for epoch in range(num_epochs):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            count =0
            while not done:
                obs = torch.tensor(obs, dtype = torch.float32)
                action_probs = self.actor(obs)
                action = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[action])
                next_obs, reward, truncation, done, info = self.env.step(action)
                next_obs = torch.tensor(next_obs, dtype = torch.float32)
                reward = torch.tensor(reward, dtype = torch.float32)
                done = torch.tensor(done, dtype = torch.float32)
                # compute critic val
                value = self.critic(obs)
                next_value = self.critic(next_obs)
                # advantage func
                advantage = reward + (1-done)*gamma*next_value - value
                advantage = advantage.detach()
                # update actor
                actor_loss  = -log_prob * advantage
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # update critic
                critic_loss = advantage.pow(2)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step() 
                count+=1
                
                obs = next_obs
                total_reward += reward.item()
                
            print(f"Epoch {epoch + 1}/{num_epochs}, Total Reward: {total_reward/count:.2f}")

        
# init
env = gym.make("LunarLander-v3")
a2c_agent = A2C(env)
a2c_agent.train(batch_size, num_epochs)
torch.save(a2c_agent.actor.state_dict(), 'Actor.pth')
torch.save(a2c_agent.critic.state_dict(), 'Critic.pth')
env.close()



