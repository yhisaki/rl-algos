import gym
import numpy as np


class DummyVectorEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = 1

    def reset(self):
        observation = self.env.reset()
        return np.array([observation])

    def step(self, action):
        observation, reward, done, info = self.env.step(action[0])
        if done:
            observation = self.env.reset()
        return np.array([observation]), np.array([reward]), np.array([done]), info
