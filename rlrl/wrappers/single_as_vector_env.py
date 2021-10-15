import gym
import numpy as np
from gym.vector import VectorEnv


class SingleAsVectorEnv(VectorEnv):
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        super().__init__(1, env.observation_space, env.action_space)

    def seed(self, seeds=None):
        if seeds is not None:
            self.env.seed(seeds[0])

    def step_async(self, actions) -> None:
        self._action = actions

    def step_wait(self):
        observation, reward, done, info = self.env.step(self._action[0])
        if done:
            observation = self.env.reset()
        return np.array([observation]), np.array([reward]), np.array([done]), [info]

    def reset_wait(self):
        observation = self.env.reset()
        return np.array([observation])

    def close_extras(self, **kwargs):
        self.env.close()
