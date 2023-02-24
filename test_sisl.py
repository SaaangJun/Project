import pettingzoo
from pettingzoo.sisl import pursuit_v4
from time import sleep
import numpy as np

env = pursuit_v4.env(render_mode='human')

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    env.step(np.random.randint(5))
    print(agent)
    env.render()
    sleep(2)

