from typing import Any, Callable, Optional
import gym
from collections.abc import Iterator


class GymInteractions(Iterator):
    """
    Examples:
    >>> env = gym.make("Swimmer-v2")
    >>> def random_actor(state):
    ...     return env.action_space.sample()
    >>> interactions = GymInteractions(env, random_actor, max_step=10000)
    >>> for step, state, next_state, action, reward, done in interactions:
    ...     ...
    """

    def __init__(
        self,
        environment: gym.Env,
        actor: Callable[[Any], Any],
        max_step: Optional[int] = None,
        max_episode: Optional[int] = None,
    ) -> None:
        """[summary]

        Args:
            environment (gym.Env): OpenAI Gym Environment
            actor (Callable[[Any], Any]): it's takes current state and returns action
            max_step (Optional[int], optional): [description]. Defaults to None.
            max_episode (Optional[int], optional): [description]. Defaults to None.
            record_reward_sum (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.env = environment
        self.actor = actor
        self.max_step = max_step
        self.max_episode = max_episode
        assert (max_step is None) ^ (max_episode is None)

        self.total_step = 0
        self.total_episode = 0

        self.episode_step = 0
        self.state = self.env.reset()
        self.done = False

        self.reward_sum = 0

    def is_finishe(self) -> bool:
        if self.max_step is not None:
            return self.max_step <= self.total_step
        elif self.max_episode is not None:
            return self.max_episode <= self.total_episode

    def __next__(self):
        if self.done:
            self.state = self.env.reset()
            self.done = False
            self.episode_step = 0
            self.reward_sum = 0

        if self.is_finishe():
            raise StopIteration()

        state = self.state
        action = self.actor(state)
        self.state, reward, self.done, _ = self.env.step(action)
        if self.done:
            self.total_episode += 1
        self.episode_step += 1
        self.total_step += 1
        self.reward_sum += reward
        return self.episode_step, state, self.state, action, reward, self.done
