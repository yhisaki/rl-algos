import gym
import numpy as np
from gym.core import Env
from gym.wrappers import TimeLimit

from rlrl.wrappers.utils import remove_wrapper


class ResetCostWrapper(gym.Wrapper):
    def __init__(self, env: Env, reset_cost: float = 100.0, terminal_step: int = 1000):
        env = remove_wrapper(env, TimeLimit)
        super().__init__(env)
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = terminal_step
        self._max_episode_steps = terminal_step
        self._reset_cost = reset_cost
        self._elapsed_steps = None
        self._reset_next_step = False
        self._reward_type = None

    def reset(self, **kwarags):
        self._elapsed_steps = 0
        self._reset_next_step = False
        return self.env.reset(**kwarags)

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        if self._reset_next_step:
            observation = self.env.reset()
            reward = -np.float_(self._reset_cost).astype(self._reward_type)
            done = False
            info = dict(is_reset_step=True)
            self._reset_next_step = False
        else:
            observation, reward, done, info = self.env.step(action)
            if self._reward_type is None:  # executed only when step() is called for the first time.
                self._reward_type = reward.dtype
            if done and (self._elapsed_steps < self._max_episode_steps):
                info["is_terminal_state"] = True
                done = False
                self._reset_next_step = True

        self._elapsed_steps += 1

        return observation, reward, (self._elapsed_steps >= self._max_episode_steps), info
