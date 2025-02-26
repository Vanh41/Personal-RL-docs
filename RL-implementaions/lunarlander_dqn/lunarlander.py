from graphviz import render
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
#
# Hyperparameters
gamma = 0.99
learning_rate = 0.001
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500
memory_size = 10000
episodes = 1000


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
    def forward(self, x):
        return self.network(x)


env = gym.make("LunarLander-v3")
q_network = QNetwork(env)
target_network = QNetwork(env)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
replay_buffer = deque(maxlen=memory_size)


def selectAction(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return q_network(state_tensor).argmax().item()



# for episode in range(episodes):
#     state, _ = env.reset()
#     episode_reward = 0
#     done = False
#     epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay)
    
#     while not done:
#         action = selectAction(state, epsilon)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         replay_buffer.append((state, action, reward, next_state, done))
#         state = next_state
#         episode_reward += reward
        
#         if len(replay_buffer) >= batch_size:
#             batch = random.sample(replay_buffer, batch_size)
#             states, actions, rewards, next_states, dones = zip(*batch)
#             states = torch.FloatTensor(states)
#             actions = torch.LongTensor(actions).unsqueeze(1)
#             rewards = torch.FloatTensor(rewards)
#             next_states = torch.FloatTensor(next_states)
#             dones = torch.FloatTensor(dones)
            
#             q_values = q_network(states).gather(1, actions).squeeze()
#             max_next_q_values = target_network(next_states).max(1)[0]
#             target = rewards + gamma * max_next_q_values * (1-done)
            
#             loss = nn.MSELoss()(q_values, target)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#     print(f"episode {episode} reward {episode_reward}")
        
# torch.save(q_network.state_dict(), 'lunar_lander_dqn.pth')
# env.close()



q_network.load_state_dict(torch.load('lunar_lander_dqn.pth'))
q_network.eval()  


env = gym.make("LunarLander-v3", render_mode = "human")
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    env.render()  
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  
        action = q_network(state_tensor).argmax().item()  
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state
print(f"Total Reward: {total_reward}")
env.close()
        
        
    
    

    


        
    