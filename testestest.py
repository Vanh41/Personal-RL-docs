import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hyperparameters
IMG_SHAPE = (3, 84, 84)  # PyTorch format: (channels, height, width)
GAMMA = 0.99
LEARNING_RATE = 3e-4
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 64
BUFFER_CAPACITY = 2048

# Preprocessing function
def preprocess_frame(frame):
    frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
    frame = torch.nn.functional.interpolate(frame.unsqueeze(0), size=(84, 84)).squeeze(0)
    return frame

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, action_space):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, action_space)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# Buffer for PPO
class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

# Calculate discounted returns
def compute_returns(rewards, dones, gamma):
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0
        G = reward + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

# PPO Training Function
def train_ppo(env, episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_space = env.action_space.n

    policy = PolicyNetwork(action_space).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    buffer = PPOBuffer()

    for episode in range(episodes):
        state = preprocess_frame(env.reset()).to(device)
        episode_reward = 0
        done = False

        while not done:
            logits, value = policy(state.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.cpu().item())
            next_state = preprocess_frame(next_state).to(device)

            buffer.store(state, action, log_prob, reward, done)
            state = next_state
            episode_reward += reward

            if len(buffer.rewards) >= BUFFER_CAPACITY or done:
                states = torch.stack(buffer.states).to(device)
                actions = torch.tensor(buffer.actions).to(device)
                log_probs = torch.stack(buffer.log_probs).to(device)
                returns = compute_returns(buffer.rewards, buffer.dones, GAMMA).to(device)

                advantages = returns - torch.tensor([policy(state.unsqueeze(0))[1] for state in buffer.states]).to(device).squeeze()

                for _ in range(UPDATE_EPOCHS):
                    indices = np.arange(len(buffer.rewards))
                    np.random.shuffle(indices)
                    for start in range(0, len(buffer.rewards), MINIBATCH_SIZE):
                        end = start + MINIBATCH_SIZE
                        mb_indices = indices[start:end]

                        mb_states = states[mb_indices]
                        mb_actions = actions[mb_indices]
                        mb_log_probs = log_probs[mb_indices]
                        mb_advantages = advantages[mb_indices]
                        mb_returns = returns[mb_indices]

                        new_logits, new_values = policy(mb_states)
                        new_dist = torch.distributions.Categorical(logits=new_logits)
                        new_log_probs = new_dist.log_prob(mb_actions)

                        ratio = torch.exp(new_log_probs - mb_log_probs)
                        surrogate1 = ratio * mb_advantages
                        surrogate2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * mb_advantages
                        actor_loss = -torch.min(surrogate1, surrogate2).mean()

                        critic_loss = (mb_returns - new_values.squeeze()).pow(2).mean()

                        loss = actor_loss + 0.5 * critic_loss - 0.01 * new_dist.entropy().mean()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                buffer.clear()

        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    return policy

# Main function
def main():
    env = gym.make('LunarLander-v3')
    episodes = 1000
    trained_policy = train_ppo(env, episodes)
    torch.save(trained_policy.state_dict(), 'lunar_lander_ppo.pth')

if __name__ == "__main__":
    main()
