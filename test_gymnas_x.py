import pettingzoo
from Project_frozenlake import gymnas_x
from time import sleep
import numpy as np

env = gymnas_x.env(render_mode='human')

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    env.step(np.random.randint(5))
    print(agent)
    env.render()
    sleep(2)

