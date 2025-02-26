import ale_py
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

rewards = []
gym.register_envs(ale_py)
# Define the Agent class
class MyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Replay buffer
        self.replay_buffer = deque(maxlen=50000)

        # Agent parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10

        # Initialize networks
        self.main_network = NeuralNetwork(state_size, action_size)
        self.target_network = NeuralNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)

        # Synchronize target network
        self.update_target_network()
        
        
        

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):
        exp_batch = random.sample(self.replay_buffer, batch_size)
        state_batch  = torch.tensor([batch[0] for batch in exp_batch], dtype=torch.float32)
        action_batch = torch.tensor([batch[1] for batch in exp_batch], dtype=torch.long)
        reward_batch = torch.tensor([batch[2] for batch in exp_batch], dtype=torch.float32)
        next_state_batch = torch.tensor([batch[3] for batch in exp_batch], dtype=torch.float32)
        terminal_batch = torch.tensor([batch[4] for batch in exp_batch], dtype=torch.float32)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def train_main_network(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)

        # Current Q values
        q_values = self.main_network(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Max Q values for the next states from the target network
        with torch.no_grad():
            max_next_q = self.target_network(next_state_batch).max(1)[0]

        # Compute target Q values
        target_q_values = reward_batch + (1 - terminal_batch) * self.gamma * max_next_q

        # Compute loss and update main network
        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_decision(self, state):
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.main_network(state)
        return torch.argmax(q_values).item()

# Main Program

# Initialize environment
env = gym.make("ALE/Pong-ram-v5")
state, _ = env.reset()

# Define state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Other parameters
n_episodes = 1000
n_timesteps = 500
batch_size = 64

# Initialize agent
my_agent = MyAgent(state_size, action_size)
total_time_step = 0

for ep in range(n_episodes):
    ep_rewards = 0
    state, _ = env.reset()

    for t in range(n_timesteps):
        total_time_step += 1

        # Update the target network periodically
        if total_time_step % my_agent.update_targetnn_rate == 0:
            my_agent.update_target_network()

        action = my_agent.make_decision(state)
        next_state, reward, terminal, _, _ = env.step(action)
        my_agent.save_experience(state, action, reward, next_state, terminal)

        state = next_state
        ep_rewards += reward
        
        if terminal:
            print("Ep ", ep+1, " reach terminal with reward = ", ep_rewards)
            rewards.append(ep_rewards)
            break

        if len(my_agent.replay_buffer) > batch_size:
            my_agent.train_main_network(batch_size)

    if my_agent.epsilon > my_agent.epsilon_min:
        my_agent.epsilon *= my_agent.epsilon_decay

# Save model weights
torch.save(my_agent.main_network.state_dict(), "train_agent_1000eps_3layers.pth")
