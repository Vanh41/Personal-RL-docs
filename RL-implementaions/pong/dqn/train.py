import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, terminal):
        self.memory.append((obs, action, next_obs, reward, terminal))

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))
    
class NeuralNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape[0]
        self.network = nn.Sequential(
            nn.Conv2d(obs_shape, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)
    
class Agent():
    def __init__(self, env):
        self.replay_buffer = ReplayBuffer(200000)  # Added capacity for ReplayBuffer
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10000  # Adjusted update rate
        self.main_network = NeuralNetwork(env)
        self.target_network = NeuralNetwork(env)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
    
    def train(self, episodes, BATCH_SIZE):
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            while not done:
                action = self.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.replay_buffer.push(state, action, next_state, reward, done)
                state = next_state
                episode_reward += reward
                
                if len(self.replay_buffer) >= BATCH_SIZE:
                    self.update_network(BATCH_SIZE)
                
                if done:
                    print(f"Episode {episode} reward: {episode_reward}")
                    break
            
            if episode % self.update_targetnn_rate == 0:
                self.update_target_network()
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.main_network(state)
            return q_values.max(1)[1].item()
    
    def update_network(self, BATCH_SIZE):
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self.replay_buffer.sample(BATCH_SIZE)
        
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.int64)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
        terminal_batch = torch.tensor(terminal_batch, dtype=torch.float32)
        
        q_values = self.main_network(state_batch)
        next_q_values = self.target_network(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - terminal_batch)
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        
EPISODES = 1000
BATCH_SIZE = 32

env = gym.make("ALE/Pong-v5")
agent = Agent(env)


agent.train(EPISODES, BATCH_SIZE)
agent.save_model("dqn_pong_model.pth")

    



