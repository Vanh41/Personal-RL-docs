import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from collections import namedtuple
from memory import RolloutBuffer
from model import ActorNetwork, CriticNetwork
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
env = gym.make("LunarLander-v3")

# hyperparameters

gamma = 0.99
learning_rate = 0.001
eps = np.finfo(np.float32).eps.item()
actor = ActorNetwork(env)
actor_optimizer = optim.Adam(actor.parameters(), lr = learning_rate)
critic = CriticNetwork(env)
critic_optimizer = optim.Adam(critic.parameters(), lr = learning_rate)
# memory = RolloutBuffer()
maxsteps = 200


# Assuming actor, critic, env, and SavedAction are defined

class Memory:
    def __init__(self):
        self.rewards = []
        self.saved_actions = []

    def clear(self):
        del self.rewards[:]
        del self.saved_actions[:]

memory = Memory()
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item()

def finish_episode():
    R = 0
    saved_actions = memory.saved_actions
    policy_losses = []
    value_losses = []
    returns = []
    for r in memory.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    # normalization
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)
        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(nn.functional.smooth_l1_loss(value, torch.tensor([R])))
    # reset gradients
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # backprop
    loss.backward()
    actor_optimizer.step()
    critic_optimizer.step()
    # reset rewards and action buffer
    memory.clear()

def main():
    num_episodes = 1500
    running_reward = 0

    for i_episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        for i in range(10000):
            state = torch.FloatTensor(state)  # Convert state to torch.Tensor
            action_probs = actor(state)
            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
            memory.saved_actions.append((dist.log_prob(action), critic(state)))
            state, reward, done, _, _ = env.step(action.item())
            memory.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # perform backprop
        finish_episode()
        # log results
        print(f"Episode {i_episode + 1}/{num_episodes}, Total Reward: {ep_reward}")

    # Save the trained models
    torch.save(actor.state_dict(), '/Users/vietanh/Documents/Personal-RL-docs/RL-implementaions/lunarlander_a2c/actor_model.pth')
    torch.save(critic.state_dict(), '/Users/vietanh/Documents/Personal-RL-docs/RL-implementaions/lunarlander_a2c/critic_model.pth')

if __name__ == '__main__':
    main()
        

        
        
    


    
    

    
    

    
    