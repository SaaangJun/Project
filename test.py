import gym
import pressureplate
from time import sleep
import numpy as np

env = gym.make('pressureplate-linear-4p-v0')
# env = gym.make('pressureplate-linear-5p-v0')
# env = gym.make('pressureplate-linear-6p-v0')

obs = env.reset()
while True:
    action = [np.random.randint(5) for _ in range(4)]
    obs, reward, done, info = env.step(action)
    sleep(0.5)
    env.render(mode='human')