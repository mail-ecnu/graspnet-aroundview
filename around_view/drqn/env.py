import os
import sys
import random
import numpy as np

import gym
from gym import spaces

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from around_view.utils.view_find import VIEW_LEN


class EnvAroundViewGrasp(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.action_space = spaces.Discrete(VIEW_LEN)
        self.observation_space = None

    def reset(self):
        return self.state

    def step(self, action):
        reward = -1
        done = -1
        return self.state, reward, done, {}

    def get_obs(self):
        return 0

    def get_global_obs(self):
        return 0

    def render(self):
        # print('agent at', env.agt_pos, 'dest', env.dest_pos, 'wrong', env.wrong_pos)
        pass


if __name__ == '__main__':
    env = EnvAroundViewGrasp(4, random.randint(0, 1))
    max_view = 5
    for i in range(max_view):
        print("iter = ", i)
        env.render()
        action = random.randint(0, 3)
        reward = env.step(action)
        print('reward', reward)
        if reward != 0:
            print('reset')
