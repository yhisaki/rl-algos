import gym
from gym.wrappers import TimeLimit

from rlrl.wrappers.utils import remove_wrapper


class ResetCostWrapper(gym.Wrapper):
    def __init__(self, env, reset_cost: float = -100.0, terminal_step: int = 1000):
        env = remove_wrapper(env, TimeLimit)
        super().__init__(env)
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = terminal_step
        self._max_episode_steps = terminal_step
        self._reset_cost = reset_cost
        self._elapsed_steps = None

    def reset(self):
        self._elapsed_steps = 0
        return super().reset()

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if done & (self._elapsed_steps < self._max_episode_steps):
            reward_type = reward.dtype
            reward += self._reset_cost
            reward = reward.astype(reward_type)
            self.env.reset()
            done = False
        elif self._elapsed_steps >= self._max_episode_steps:
            done = True
        return observation, reward, done, info


if __name__ == "__main__":
    from rlrl.wrappers import make_env

    env = make_env("Hopper-v3")
    print(env.unwrapped)
