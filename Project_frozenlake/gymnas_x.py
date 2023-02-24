# Dongjae Kim gymnasium environment
# Importing libraries
from collections import defaultdict
from gymnasium.utils import EzPickle
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn


import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import pygame
from pettingzoo import AECEnv

# copied from sisl env from pettingzoo.
from .utils import agent_utils, two_d_maps
from .utils.agent_layer import AgentLayer
from .utils.controllers import PursuitPolicy, RandomPolicy, SingleActionPolicy


import numpy as np  # To deal with data in form of matrices
import tkinter as tk  # To build GUI
import time  # Time is needed to slow down the agent and to see how he runs
from PIL import Image, ImageTk  # For adding images into the canvas widget

# Global variable for dictionary with coordinates for the final route
a = {}

class coEnv_base():
    def __init__(
        self,
        x_size: int = 25,
        y_size: int = 25,
        max_cycles: int = 500,
        shared_reward: bool = True,
        n_agents: int = 3,
        obs_range: int = 7,
        surround: bool = True,
        render_mode=None,
        constraint_window: float = 1.0,
        p_obstacle: float = 0.0
    ):
        self.x_size = x_size
        self.y_size = y_size
        self.map_matrix = two_d_maps.random_obstacle_map(self.x_size-2, self.y_size-2, p_obstacle)
        self.max_cycles = max_cycles
        self.seed()
        self.goal_matrix = self.map_matrix.copy() # initialize it first.

        self.shared_reward = shared_reward
        self.local_ratio = 1.0 - float(self.shared_reward)

        self.num_agents = n_agents

        self.latest_reward_state = [0 for _ in range(self.num_agents)]
        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.latest_obs = [None for _ in range(self.num_agents)]

        # can see 7 grids around them by default
        self.obs_range = obs_range
        # assert self.obs_range % 2 != 0, "obs_range should be odd"
        self.obs_offset = int((self.obs_range - 1) / 2)
        self.agents = agent_utils.create_agents(
            self.num_agents, self.map_matrix, self.obs_range, self.np_random
        )
        self.agent_layer = AgentLayer(x_size, y_size, self.agents)


        n_act_agents = self.agent_layer.get_nactions(0)


        self.agent_controller = (
            RandomPolicy(n_act_agents, self.np_random)
        )
        self.current_agent_layer = np.zeros((x_size, y_size), dtype=np.int32)
        self._reward = 1

        self.ally_actions = np.zeros(n_act_agents, dtype=np.int32)

        max_agents_overlap = self.num_agents
        obs_space = spaces.Box(
            low=0,
            high=max_agents_overlap,
            shape=(self.obs_range, self.obs_range, 3),
            dtype=np.float32,
        )

        act_space = spaces.Discrete(n_act_agents)
        self.action_space = [act_space for _ in range(self.num_agents)]
        self.observation_space = [obs_space for _ in range(self.num_agents)]
        self.act_dims = [n_act_agents for i in range(self.num_agents)]

        self.surround = surround

        self.render_mode = render_mode
        self.constraint_window = constraint_window

        self.surround_mask = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

        self.model_state = np.zeros((1+self.num_agents,) + self.map_matrix.shape, dtype=np.float32)
        # TODO check if the above one makes sense

        self.renderOn = False
        self.pixel_scale = 30

        self.frames = 0
        self.reset()
    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        try:
            policies = [self.agent_controller]
            for policy in policies:
                try:
                    policy.set_rng(self.np_random)
                except AttributeError:
                    pass
        except AttributeError:
            pass

        return [seed_]
    def set_goal_matrix(self):
        empty_ind = np.where(self.goal_matrix==0)
        empty_ind = [empty_ind[0].tolist(),empty_ind[1].tolist()]
        for agent in range(self.num_agents):
            ind = np.random.randint(len(empty_ind[0]))
            self.goal_matrix[empty_ind[0].pop(ind),empty_ind[1].pop(ind)] = agent
        return self.goal_matrix
    def reset(self):
        # x_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        # y_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        # xlb, xub = int(self.x_size * x_window_start), int(
        #     self.x_size * (x_window_start + self.constraint_window)
        # )
        # ylb, yub = int(self.y_size * y_window_start), int(
        #     self.y_size * (y_window_start + self.constraint_window)
        # )
        # constraints = [[xlb, xub], [ylb, yub]]
        #
        # self.agents = agent_utils.create_agents(
        #     self.num_agents,
        #     self.map_matrix,
        #     self.obs_range,
        #     self.np_random,
        #     randinit=True,
        #     constraints=constraints,
        # )

        self.model_state[0] = self.map_matrix
        # TODO check whether it makes sense
        self.goal_matrix = self.map_matrix.copy()
        self.model_state[2] = self.set_goal_matrix()

        self.agents = agent_utils.create_agents_by_goals(
            self.num_agents,  self.model_state, self.obs_range, self.np_random
        )

        self.agent_layer = AgentLayer(self.x_size, self.y_size, self.agents)
        self.latest_reward_state = [0 for _ in range(self.num_agents)]
        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.latest_obs = [None for _ in range(self.num_agents)]

        self.model_state[1] = self.agent_layer.get_state_matrix()

        self.frames = 0
        self.renderOn = False

        return self.safely_observe(0)

    def safely_observe(self, i):
        agent_layer = self.agent_layer
        obs = self.collect_obs(agent_layer, i)
        return obs

    def collect_obs(self, agent_layer, i):
        for j in range(self.n_agents()):
            if i == j:
                return self.collect_obs_by_idx(agent_layer, i)
        assert False, "bad index"

    def collect_obs_by_idx(self, agent_layer, agent_idx):
        # returns a flattened array of all the observations
        obs = np.zeros((3, self.obs_range, self.obs_range), dtype=np.float32)
        obs[0].fill(1.0)  # border walls set to -0.1?
        xp, yp = agent_layer.get_position(agent_idx)

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

        obs[0:2, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:2, xlo:xhi, ylo:yhi]) # TODO check. originally np.abs(self.model_state[0:3, xlo:xhi, ylo:yhi])
        return obs

    def obs_clip(self, x, y):
        xld = x - self.obs_offset
        xhd = x + self.obs_offset
        yld = y - self.obs_offset
        yhd = y + self.obs_offset
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_size - 1),
            np.clip(xhd, 0, self.x_size - 1),
            np.clip(yld, 0, self.y_size - 1),
            np.clip(yhd, 0, self.y_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.obs_offset, 0)), abs(
            np.clip(yld, -self.obs_offset, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def n_agents(self):
        return self.agent_layer.n_agents()

    def step(self, action, agent_id, is_last):
        agent_layer = self.agent_layer

        # actual action application, change the pursuer layer
        agent_layer.move_agent(agent_id, action)

        # TODO: 여기서 goal condition에 대해서 추가해야할 수 있다.
        # Update agent layer
        self.model_state[1] = self.agent_layer.get_state_matrix()
        self.latest_reward_state = self.reward() / self.num_agents

        # Update the remaining layers
        self.model_state[0] = self.map_matrix

        global_val = self.latest_reward_state.mean()
        local_val = self.latest_reward_state
        self.latest_reward_state = (
            self.local_ratio * local_val + (1 - self.local_ratio) * global_val
        )

        if self.render_mode == "human":
            self.render()

    # TODO reward(self) needs to be updated to get reward when the success condition reached.
    def reward(self):
        rewards = np.ones((self.num_agents))
        rewards = rewards*self._reward
        return rewards

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if not self.renderOn:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )
            else:
                self.screen = pygame.Surface(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )

            self.renderOn = True
        self.draw_model_state()
        self.draw_agents_observations()

        self.draw_agents()
        self.draw_agent_counts()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_mode == "human":
            pygame.display.flip()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )


    def draw_model_state(self):
        # -1 is building pixel flag
        x_len, y_len = self.model_state[0].shape
        for x in range(x_len):
            for y in range(y_len):
                pos = pygame.Rect(
                    self.pixel_scale * x,
                    self.pixel_scale * y,
                    self.pixel_scale,
                    self.pixel_scale,
                )
                col = (0, 0, 0)
                if self.model_state[0][x][y] == -1:
                    col = (255, 255, 255)
                pygame.draw.rect(self.screen, col, pos)

    def draw_agents_observations(self):
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            patch = pygame.Surface(
                (self.pixel_scale * self.obs_range, self.pixel_scale * self.obs_range)
            )
            patch.set_alpha(128)
            patch.fill((255, 152, 72))
            ofst = self.obs_range / 2.0
            self.screen.blit(
                patch,
                (
                    self.pixel_scale * (x - ofst + 1 / 2),
                    self.pixel_scale * (y - ofst + 1 / 2),
                ),
            )

    def draw_agents(self):
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (255, 0, 0)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))

    def draw_agent_counts(self):
        font = pygame.font.SysFont("Comic Sans MS", self.pixel_scale * 2 // 3)

        agent_positions = defaultdict(int)
        evader_positions = defaultdict(int)

        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            agent_positions[(x, y)] += 1

        for (x, y) in evader_positions:
            (pos_x, pos_y) = (
                self.pixel_scale * x + self.pixel_scale // 2,
                self.pixel_scale * y + self.pixel_scale // 2,
            )

            agent_count = evader_positions[(x, y)]
            count_text: str
            if agent_count < 1:
                count_text = ""
            elif agent_count < 10:
                count_text = str(agent_count)
            else:
                count_text = "+"

            text = font.render(count_text, False, (0, 255, 255))

            self.screen.blit(text, (pos_x, pos_y))

        for (x, y) in agent_positions:
            (pos_x, pos_y) = (
                self.pixel_scale * x + self.pixel_scale // 2,
                self.pixel_scale * y + self.pixel_scale // 2,
            )

            agent_count = agent_positions[(x, y)]
            count_text: str
            if agent_count < 1:
                count_text = ""
            elif agent_count < 10:
                count_text = str(agent_count)
            else:
                count_text = "+"

            text = font.render(count_text, False, (255, 255, 0))

            self.screen.blit(text, (pos_x, pos_y - self.pixel_scale // 2))

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            pygame.quit()
            self.renderOn = False

    @property
    def is_terminal(self):
        # ev = self.evader_layer.get_state_matrix()  # evader positions
        # if np.sum(ev) == 0.0:
        # TODO end conditions
        if self.agent.n_agents() == 0:
            return True
        return False

class coEnv(AECEnv,EzPickle): # cooperation environemnt

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "coEnv",
        "is_parallelizable": True,
        "render_fps": 5,
        "has_manual_policy": True,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        self.env = coEnv_base(*args, **kwargs)
        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        self.agents = ["Agent_" + str(a) for a in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        # spaces
        self.n_act_agents = self.env.act_dims[0]
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.steps = 0
        self.closed = False
    def seed(self, seed=None):
        self.env.seed(seed)

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents[:]
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.env.reset()

    def close(self):
        if not self.closed:
            self.closed = True
            self.env.close()

    def render(self):
        if not self.closed:
            return self.env.render()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self.env.step(
            action, self.agent_name_mapping[agent], self._agent_selector.is_last()
        )
        for k in self.terminations:
            if self.env.frames >= self.env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = self.env.is_terminal
        for k in self.agents:
            self.rewards[k] = self.env.latest_reward_state[self.agent_name_mapping[k]]
        self.steps += 1

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def observe(self, agent):
        o = self.env.safely_observe(self.agent_name_mapping[agent])
        return np.swapaxes(o, 2, 0)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]



# Returning the final dictionary with route coordinates
# Then it will be used in agent_brain.py
def final_states():
    return a


def env(**kwargs):
    env = coEnv(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


# This we need to debug the environment
# If we want to run and see the environment without running full algorithm
if __name__ == '__main__':

    from time import sleep
    env = env(render_mode='human')

    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        env.step(np.random.randint(5))
        print(agent)
        env.render()
        sleep(2)
