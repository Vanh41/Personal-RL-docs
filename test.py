import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical

# Actor Network
class Actor(nn.Module):
    def __init__(self, input_size, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_model(actor, critic, actor_path, critic_path):
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    print(f"Saved actor model to {actor_path}")
    print(f"Saved critic model to {critic_path}")

def load_model(actor, critic, actor_path, critic_path):
    actor.load_state_dict(torch.load(actor_path))
    critic.load_state_dict(torch.load(critic_path))
    actor.eval()
    critic.eval()
    print(f"Loaded actor model from {actor_path}")
    print(f"Loaded critic model from {critic_path}")

def actor_critic(actor, critic, episodes, max_steps=2000, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3):
    optimizer_actor = optim.AdamW(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=lr_critic)
    stats = {'Actor Loss': [], 'Critic Loss': [], 'Returns': []}

    env = gym.make('LunarLander-v3')
    input_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    for episode in range(1, episodes + 1):
        state, info = env.reset()
        ep_return = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            state_tensor = torch.FloatTensor(state)
            
            # Actor selects action
            action_probs = actor(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # Take action and observe next state and reward
            next_state, reward, terminated, done, info = env.step(action.item())
            next_state_tensor = torch.FloatTensor(next_state)
            ep_return += reward
            step_count += 1

            # Update state
            state = next_state

        stats['Returns'].append(ep_return)
        print(f"Episode {episode}/{episodes}, Return: {ep_return}")

    return stats

if __name__ == "__main__":
    actor_path = "actor_model.pth"
    critic_path = "critic_model.pth"

    env = gym.make('LunarLander-v3')
    input_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Check if pretrained model exists
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        print("Loading pretrained models...")
        actor = Actor(input_size=input_size, num_actions=num_actions)
        critic = Critic(input_size=input_size)
        load_model(actor, critic, actor_path, critic_path)
    else:
        print("Training new models...")
        actor = Actor(input_size=input_size, num_actions=num_actions)
        critic = Critic(input_size=input_size)
        episodes = 100  # Number of episodes for training
        stats = actor_critic(actor, critic, episodes)
        save_model(actor, critic, actor_path, critic_path)
    
    # Test the trained agent in human mode
    env = gym.make('LunarLander-v3', render_mode='human')
    state = env.reset()[0]
    done = False
    total_reward = 0
    max_steps = 2000  # Maximum steps per episode for testing
    
    while not done:
        env.render()
        state_tensor = torch.FloatTensor(state)
        action_probs = actor(state_tensor)
        action = torch.argmax(action_probs).item()
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if total_reward >= max_steps:
            break
    
    print(f"Total reward in human mode: {total_reward}")
    env.close()