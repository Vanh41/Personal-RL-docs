import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    
    def __init__(self, env,
                 features_dim: int = 512,
                 ):
        super().__init__()
        n_input_channels = env.observation_space.shape[0]
        self.actor = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.critic = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.actor(torch.as_tensor(env.observation_space.sample()[None]).float()).shape[1]
            
        self.linearActor = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.Tanh())
        self.linearCritic = nn.Sequential(nn.Linear(n_flatten, features_dim), 1)
        
    def forwardActor(self, observation: torch.Tensor) -> torch.Tensor:
        return self.linearActor(self.actor(observation))
    
    def forwardCritic(self, observation: torch.Tensor) -> torch.Tensor:
        return self.linearCritic(self.critic(observation))
    
    
    

            
        
        
    
    