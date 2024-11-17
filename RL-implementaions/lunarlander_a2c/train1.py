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
maxsteps = 200

# cal return
def calculate_returns(rewards, normalize = True):
    returns = []
    reward = 0
    for r in reversed(rewards):
        reward = r + reward * gamma
        returns.insert(0, reward)        
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

# cal adv
def calculate_advantages(returns, values, normalize = True):
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

# update policy
def update_policy(advantages, log_prob_actions, returns, values):   
    advantages = advantages.detach()
    returns = returns.detach()
    actor_loss = - (advantages * log_prob_actions).sum()
    critic_loss = nn.functional.smooth_l1_loss(returns, values).sum()
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    actor_optimizer.step()
    critic_optimizer.step()
    return actor_loss.item(), critic_loss.item()

def evaluate():
    actor.eval()
    critic.eval()
    rewards = []
    done = False
    episode_reward = 0
    state = env.reset()
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_prob = actor(state)
        action = torch.argmax(action_prob, dim = -1)
        state, reward, done, _ = env.step(action.item())
        episode_reward += reward
    return episode_reward

# train func
def train():
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    state, info = env.reset()
    while not done:
        state = torch.FloatTensor(state)
        # actor select action
        action_prob = actor(state)
        dist = torch.distributions.Categorical(probs = action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        state, reward, done, _, _ = env.step(action.item())
        # critic predict value
        value = critic(state)
        # save to mem
        log_prob_action = dist.log_prob(action)
        log_prob_actions.append(log_prob_action)
        values.append(value)
        rewards.append(reward)
        episode_reward+=reward
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, gamma)
    advantages = calculate_advantages(returns, values)
    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values)
    return policy_loss, value_loss, episode_reward


MAX_EPISODES = 1_000
N_TRIALS = 25
REWARD_THRESHOLD = 200
PRINT_EVERY = 10

train_rewards = []
test_rewards = []

for episode in range(1, MAX_EPISODES+1):
    policy_loss, value_loss, train_reward = train()
    test_reward = evaluate()
    train_rewards.append(train_reward)
    test_rewards.append(test_reward)
    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
    if episode % PRINT_EVERY == 0:
        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')
    if mean_test_rewards >= REWARD_THRESHOLD:
        print(f'Reached reward threshold in {episode} episodes')
        break
    

torch.save(actor.state_dict(), 'actor_model.pth')
torch.save(critic.state_dict(), 'critic_model.pth')
        
    
    