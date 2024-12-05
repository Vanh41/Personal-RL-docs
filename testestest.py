import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt

env = gym.make("MiniGrid-SimpleCrossingS11N5-v0", render_mode = 'rgb_array')

state, _ = env.reset()
frame = env.render()

plt.imshow(frame)
plt.axis('off') 
plt.show()