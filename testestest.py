import gymnasium as gym
import minigrid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# Tạo môi trường MiniGrid
env = gym.make("MiniGrid-LavaCrossingS11N5-v0")

# Cấu trúc mạng Actor-Critic
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCriticNetwork, self).__init__()
        
        # Mạng CNN để xử lý ảnh
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Lớp Flatten
        self.flatten = nn.Flatten()
        
        # Lớp fully connected
        self.fc = nn.Linear(128 * 11 * 11, 512)
        
        # Actor: Lớp output phân phối xác suất cho hành động
        self.actor_fc = nn.Linear(512, num_actions)
        
        # Critic: Lớp output giá trị (value function)
        self.critic_fc = nn.Linear(512, 1)

    def forward(self, x):
        # Pass qua các lớp CNN
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Lớp Flatten
        x = self.flatten(x)
        
        # Pass qua lớp fully connected
        x = torch.relu(self.fc(x))
        
        # Output của actor (phân phối xác suất hành động)
        action_probs = torch.softmax(self.actor_fc(x), dim=-1)
        
        # Output của critic (giá trị trạng thái)
        state_value = self.critic_fc(x)
        
        return action_probs, state_value

# Khởi tạo môi trường và mạng
input_shape = (3, 11, 11)  # 3 channel RGB, 11x11 grid
num_actions = env.action_space.n
actor_critic = ActorCriticNetwork(input_shape, num_actions)

# Khởi tạo optimizer
optimizer = optim.Adam(actor_critic.parameters(), lr=0.0003)

# Hàm tính lợi nhuận (return)
def compute_returns(next_value, rewards, masks, gamma=0.99):
    returns = []
    R = next_value
    for reward, mask in zip(reversed(rewards), reversed(masks)):
        R = reward + gamma * R * mask
        returns.insert(0, R)
    return returns

# Hàm huấn luyện
def train(env, actor_critic, optimizer, num_epochs=1000, gamma=0.99):
    for epoch in range(num_epochs):
        state = env.reset()
        state = torch.tensor(state['image'], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2) / 255.0  # Normalize
        
        rewards = []
        log_probs = []
        values = []
        masks = []
        actions = []
        
        for t in range(1000):  # Maximum steps per episode
            # Chọn hành động từ actor
            action_probs, state_value = actor_critic(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # Thực hiện hành động
            next_state, reward, done, _, info = env.step(action.item())
            next_state = torch.tensor(next_state['image'], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2) / 255.0  # Normalize
            
            # Lưu trữ giá trị, hành động và phần thưởng
            rewards.append(reward)
            log_probs.append(dist.log_prob(action))
            values.append(state_value)
            masks.append(1 - int(done))  # done -> 1 nếu kết thúc, 0 nếu tiếp tục
            actions.append(action.item())
            
            # Cập nhật trạng thái
            state = next_state
            
            if done:
                break
        
        # Tính giá trị return (lợi nhuận)
        next_value = actor_critic(state)[1]
        returns = compute_returns(next_value, rewards, masks, gamma)
        returns = torch.tensor(returns).detach()
        
        # Tính toán loss và gradient descent
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        
        # Tính loss
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        loss = actor_loss + 0.5 * critic_loss
        
        # Cập nhật tham số mạng
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # In thông tin mỗi epoch
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Loss: {loss.item():.3f}")

# Huấn luyện
train(env, actor_critic, optimizer, num_epochs=1000)
