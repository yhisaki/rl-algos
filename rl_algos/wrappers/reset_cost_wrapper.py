from typing import Any

import gymnasium
import numpy as np
from gymnasium.core import Env


class ResetCostWrapper(gymnasium.Wrapper):
    def __init__(self, env: Env, reset_cost: float = 100.0):
        super().__init__(env)
        self._reset_cost = reset_cost
        self._reset_next_step = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._reset_next_step = False
        return super().reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        if self._reset_next_step:
            observation, info = self.env.reset()
            reward = -self._reset_cost
            terminated = False
            truncated = False
            info.update({"is_terminal_state": True})
            self._reset_next_step = False
        else:
            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated:
                terminated = False
                self._reset_next_step = True

        if self._reset_next_step:
            info.update({"is_terminal_state": True})
        else:
            info.update({"is_terminal_state": False})

        return observation, reward, False, truncated, info
