import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import ale_py

# Không cần gọi gym.register_envs(ale_py) nếu môi trường đã được đăng ký sẵn.

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
        # Giả sử observation có định dạng (height, width, channels)
        input_channels = env.observation_space.shape[2]
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # Đảm bảo kích thước đầu ra cố định
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)  # Số hành động của môi trường
        )
    
    def forward(self, x):
        return self.network(x)
    
class Agent():
    def __init__(self, env):
        self.env = env
        self.replay_buffer = ReplayBuffer(200000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_targetnn_rate = 100  # cập nhật target network mỗi 100 tập (có thể điều chỉnh)
        self.main_network = NeuralNetwork(env)
        self.target_network = NeuralNetwork(env)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
    
    def train(self, episodes, BATCH_SIZE):
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.push(state, action, next_state, reward, done)
                state = next_state
                episode_reward += reward
                
                if len(self.replay_buffer) >= BATCH_SIZE:
                    self.update_network(BATCH_SIZE)
            
            print(f"Episode {episode} reward: {episode_reward}")
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if episode % self.update_targetnn_rate == 0:
                self.update_target_network()
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # Chuyển đổi state từ định dạng HWC sang CHW
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)
            q_values = self.main_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def update_network(self, BATCH_SIZE):
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self.replay_buffer.sample(BATCH_SIZE)
        
        # Chuyển đổi sang tensor và chuyển từ HWC sang CHW
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).permute(0, 3, 1, 2)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).permute(0, 3, 1, 2)
        
        action_batch = torch.tensor(action_batch, dtype=torch.int64)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        terminal_batch = torch.tensor(terminal_batch, dtype=torch.float32)
        
        q_values = self.main_network(state_batch)
        # Lấy Q-value tương ứng với hành động đã thực hiện
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        next_q_values = self.target_network(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - terminal_batch)
        
        loss = nn.MSELoss()(q_value, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    def save_model(self, path):
        torch.save(self.main_network.state_dict(), path)
        
        
EPISODES = 1000
BATCH_SIZE = 32

env = gym.make("ALE/Pong-v5")
agent = Agent(env)

agent.train(EPISODES, BATCH_SIZE)
agent.save_model("dqn_pong_model.pth")
