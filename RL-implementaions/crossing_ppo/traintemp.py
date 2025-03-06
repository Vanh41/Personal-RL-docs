import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import random
from collections import deque
from memory import RolloutBuffer
from model import ActorNetwork, CriticNetwork
from minigrid.wrappers import FlatObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper
from torchvision import transforms



env = gym.make("MiniGrid-Empty-5x5-v0", render_mode = 'human')

# env = RGBImgPartialObsWrapper(env) # Get pixel observations
# env = ImgObsWrapper(env) # Get rid of the 'mission' field

n_frames = 10**6
n_epochs = 4
gamma = 0.99
learning_rate = 0.001
value_loss_coef = 0.5
entropy_coef = 0.01
epsilon = 1e-8
gae_lambda = 0.95
clip_epsilon = 0.2
batch_size = 10
num_agents = 16
num_runs = 30

actor = ActorNetwork(env)
critic = CriticNetwork(env)
actor_optimizer = optim.Adam(actor.parameters(), lr = learning_rate, eps = epsilon)
critic_optimizer = optim.Adam(critic.parameters(), lr = learning_rate, eps = epsilon)
buffer = RolloutBuffer(batch_size)


def chooseAction(state):
    state = torch.FloatTensor(state['image']).permute(2, 0, 1)
    if state.ndim==3:
        state = state.unsqueeze(0)
    action_probs = actor(state)
    dist = torch.distributions.Categorical(action_probs)
    action = dist.sample()
    prob = torch.squeeze(dist.log_prob(action)).item()
    value = critic(state)
    value = torch.squeeze(value).item()
    return action.item(), prob, value

def train():
    for _ in range(n_epochs):
        action_arr, state_arr, old_prob_arr, values_arr, reward_arr, dones_arr, batches = buffer.generate_batches()
        values = values_arr
        advantage = np.zeros(len(reward_arr), dtype = np.float32)
        
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k]+ gamma * values_arr[k + 1]*(1-int(dones_arr[k])) - values_arr[k])
                discount *= gae_lambda*gamma
            advantage[t] = a_t
            
        advantage = torch.FloatTensor(advantage)
        values = torch.FloatTensor(values)
        
        for batch in batches:
            states = np.stack([x['image'] for x in state_arr[batch].tolist()])
            states = torch.FloatTensor(states).permute(0,3,1,2)
       
            old_probs = torch.FloatTensor(old_prob_arr[batch])
            actions = torch.FloatTensor(action_arr[batch])
            action_probs = actor(states)
            dist = torch.distributions.Categorical(probs = action_probs)
            new_probs = dist.log_prob(actions)
            critic_value = critic(states)
            critic_value = torch.squeeze(critic_value)

            prob_ratio = torch.exp(new_probs - old_probs)
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage[batch]
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            
            returns = advantage[batch] + values[batch]
            critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss.mean()

            entropy = -torch.mean( (new_probs)* old_probs.exp())
            
            total_loss = actor_loss - value_loss_coef * critic_loss + entropy_coef * entropy
            
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            total_loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()
    buffer.clear()
                
                
if __name__ == '__main__':
    n_steps = 0
    N = 20
    learn_iters = 0
    score_history = []
    for i in range(1000):
        state, _ = env.reset()
        done = False
        score = 0 
        steps = 0 
        max_steps = 1000
        while not done and steps < max_steps:
            action, prob, val = chooseAction(state)
            next_state, reward, done, info, _ = env.step(action)
            n_steps += 1
            steps +=1
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
            
            


