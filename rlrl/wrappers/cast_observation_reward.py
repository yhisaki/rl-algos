from typing import Any

import gym
import numpy as np
from gym.spaces.box import Box

# refer to https://github.com/pfnet/pfrl/blob/master/pfrl/wrappers/cast_observation.py


class CastObservation(gym.ObservationWrapper):
    """Cast observations to a given type.

    Args:
        env: Env to wrap.
        dtype: Data type object.

    Attributes:
        original_observation: Observation before casting.
    """

    def __init__(self, env, dtype):
        super().__init__(env)
        if isinstance(self.observation_space, Box):
            self.observation_space.dtype = np.dtype(dtype)
            self.observation_space.low = self.observation_space.low.astype(dtype)
            self.observation_space.high = self.observation_space.high.astype(dtype)
        self.dtype = dtype

    def observation(self, observation):
        self.original_observation = observation
        return observation.astype(self.dtype, copy=False)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)


class CastObservationToFloat32(CastObservation):
    """Cast observations to float32, which is commonly used for NNs.

    Args:
        env: Env to wrap.

    Attributes:
        original_observation: Observation before casting.
    """

    def __init__(self, env):
        super().__init__(env, np.float32)


class CastRewardToFloat32(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        self.original_reward = reward
        return reward.astype(np.float32, copy=False)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)
