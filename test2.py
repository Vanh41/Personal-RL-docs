import gymnasium as gym 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
env = gym.make("CarRacing-v3", render_mode = "rgb_array")

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=200000, progress_bar=True)