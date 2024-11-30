import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from memory import RolloutBuffer
from model import ActorNetwork, CriticNetwork

env = gym.make("LunarLander-v3")
# hyperparamters 
gamma = 0.99
learning_rate = 0.0001
alpha = 0.0003
gae_lambda = 0.95
clip_epsilon = 0.2
batch_size = 64
n_epochs = 10
num_episodes = 1028

actor = ActorNetwork(env)
critic = CriticNetwork(env)
actor_optimizer = optim.Adam(actor.parameters(), lr = learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr = learning_rate)
critic = CriticNetwork(env)
buffer = RolloutBuffer(batch_size)

def chooseAction(state):
    state = torch.FloatTensor(state)
    action_probs = actor(state)
    dist = torch.distributions.Categorical(probs = action_probs)
    action = dist.sample()
    prob = torch.squeeze(dist.log_prob(action)).item()
    value = critic(state)
    value = torch.squeeze(value).item()
    return action.item(), prob, value


def train():
    for i in range(n_epochs):
        action_arr, state_arr, old_prob_arr, values_arr, reward_arr, dones_arr, batches = buffer.generate_batches()
        values = values_arr
        advantage = torch.zeros(len(reward_arr), dtype = torch.float32)
        
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + gamma*values[k + 1] * (1-dones_arr[k]) - values[k])
                discount *= gamma*gae_lambda
            advantage[t] = a_t
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage = torch.FloatTensor(advantage)
        values = torch.FloatTensor(values)
        for batch in batches:
            states = torch.FloatTensor(state_arr[batch])
            old_probs = torch.FloatTensor(old_prob_arr[batch])
            actions = torch.FloatTensor(action_arr[batch])
            
            action_probs = actor(states)
            dist = torch.distributions.Categorical(probs = action_probs)
            new_probs = dist.log_prob(actions)
            critic_value = critic(states)
            critic_value = torch.squeeze(critic_value)

            ratio = torch.exp(new_probs - old_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            actor_loss = -torch.min(ratio * advantage[batch], clipped_ratio * advantage[batch]).mean()
            
            returns = advantage[batch] + values[batch]
            # critic_loss = nn.MSELoss()(returns, critic_value)
            critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss.mean()
            loss = actor_loss + 0.5 * critic_loss
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()
    buffer.clear()
    


if __name__ == '__main__':
    n_steps = 0
    N = 20
    learn_iters = 0
    score_history = []
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = chooseAction(state)
            next_state, reward, done, info, _ = env.step(action)
            n_steps += 1
            buffer.store_memory(state, action, prob, val, reward, done)
            if n_steps % N == 0:
                train()
                learn_iters += 1
            score += reward
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print(f'episode {i} score {score} avg score {avg_score}')
        
torch.save(actor.state_dict(), 'actor_model.pth')
torch.save(critic.state_dict(), 'critic_model.pth')


            

        
