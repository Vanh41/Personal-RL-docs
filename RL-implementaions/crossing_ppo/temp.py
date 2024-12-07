import gym
from minigrid.wrappers import FullyObsWrapper
from stable_baselines3 import PPO

env_id = 'MiniGrid-LavaCrossingS11N5-v0'
env = FullyObsWrapper(gym.make(env_id))

model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_minigrid_lavacrossing")

# Đánh giá mô hình
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
