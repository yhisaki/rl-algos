import collections
from collections.abc import Iterator
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from gymnasium import Env
from gymnasium.vector import SyncVectorEnv, VectorEnv

from rl_algos.utils import clear_if_maxlen_is_none, logger, mean_or_nan


class TransitionGenerator(Iterator):
    """
    This class is an iterator that generates experiences from a given environment.
    It interacts with the environment using the provided actor function \
        and collects transition tuples.
    It can also calculate and provide statistics about the collected experiences.

    Args:
        environment (Union[Env, VectorEnv]): The environment to generate experiences from.
        actor (Callable[[Any], Any]): The function that determines \
            the action given the current state.
        max_step (Optional[int]): The maximum number of steps to take. If this is set, \
            max_episode must be None.
        max_episode (Optional[int]): The maximum number of episodes to generate. If this is set, \
            max_step must be None.
        calc_stats (bool): Whether to calculate statistics about the experiences.
        step_stats_window (int): The size of the window for calculating step statistics.
        reward_sum_stats_window (int): The size of the window for calculating reward sum statistics.
    """

    def __init__(
        self,
        environment: Union[Env, VectorEnv],
        actor: Callable[[Any], Any],
        max_step: Optional[int] = None,
        max_episode: Optional[int] = None,
        calc_stats: bool = True,
        step_stats_window: int = None,
        reward_sum_stats_window: int = None,
    ) -> None:
        super().__init__()
        self.env = environment
        self.actor = actor
        self.max_step = max_step
        self.max_episode = max_episode

        # Ensure that either max_step or max_episode is set, but not both
        assert (max_step is None) ^ (
            max_episode is None
        ), "Either max_episode or max_step must be set to a value."

        # If the environment is not a VectorEnv, wrap it in a SyncVectorEnv
        if not isinstance(self.env, VectorEnv):
            self.env = SyncVectorEnv([lambda: self.env])

        # Reset the environment and get the initial state
        self.state, info = self.env.reset()

        # Initialize various counters and flags
        self.num_envs: int = self.env.num_envs
        self.total_step = np.zeros(self.num_envs, np.uint)
        self.total_episode = np.zeros(self.num_envs, np.uint)
        self.episode_step = np.zeros(self.num_envs, np.uint)
        self.episode_reward = np.zeros(self.num_envs, np.float32)  # reward sum per episode
        self.terminated = np.zeros(self.num_envs, np.bool_)
        self.truncated = np.zeros(self.num_envs, np.bool_)

        # Initialize logger
        self.logger = logger.getChild(self.__class__.__name__)

        # Initialize statistics calculation if required
        self.calc_stats = calc_stats
        if calc_stats:
            self.step_record = collections.deque(maxlen=step_stats_window)
            self.reward_sum_record = collections.deque(maxlen=reward_sum_stats_window)

    def is_finish(self) -> bool:
        """
        Check if the maximum number of steps or episodes has been reached.

        Returns:
            bool: True if the maximum has been reached, False otherwise.
        """
        if self.max_step is not None:
            return self.max_step <= self.total_step.sum()
        elif self.max_episode is not None:
            return self.max_episode <= self.total_episode.sum()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, ...]:
        """
        Generate the next experience by interacting with the environment.

        Returns:
            Tuple[np.ndarray, ...]: The generated experience as a tuple of \
                (episode_step, state, next_state, action, reward, terminated, truncated, info).

        Raises:
            StopIteration: If the maximum number of steps or episodes has been reached.
        """
        # Reset episode reward and step if the episode has terminated or been truncated
        self.episode_reward *= np.invert(self.terminated | self.truncated)
        self.episode_step *= np.invert(self.terminated | self.truncated)

        # Stop iteration if the maximum number of steps or episodes has been reached
        if self.is_finish():
            raise StopIteration()

        # Copy the current state, get the action from the actor, and take a step in the environment
        state = np.copy(self.state)
        action = self.actor(state)
        next_state, reward, self.terminated, self.truncated, info = self.env.step(action)

        # Update the current state
        self.state = np.copy(next_state)

        # Check if the episode has finished
        episode_finish = self.terminated | self.truncated

        # Update total episode and step counters
        self.total_episode += episode_finish
        self.total_step += np.ones_like(self.total_episode)

        # Update episode reward and step counters
        self.episode_reward += np.nan_to_num(reward)
        self.episode_step += np.ones_like(self.episode_step)

        def process_episode_finished_environment(
            env_idx,
            episode_finish,
            episode_reward,
            episode_step,
            total_step,
            final_observation,
            next_state,
        ):
            """
            Process an environment where the episode has finished.

            Args:
                env_idx (int): The index of the environment.
                episode_finish (bool): Whether the episode has finished.
                episode_reward (float): The total reward for the episode.
                episode_step (int): The number of steps taken in the episode.
                total_step (int): The total number of steps taken across all episodes.
                final_observation (np.ndarray): The final observation from the environment.
                next_state (np.ndarray): The next state from the environment.

            Returns:
                np.ndarray: The final observation if the episode has finished, \
                    otherwise the next state.
            """
            if episode_finish:
                self.reward_sum_record.append(episode_reward)
                self.step_record.append(episode_step)
                self.logger.info(
                    msg=f"env_idx={env_idx},"
                    f"episode_step = {episode_step},"
                    f"total_step = {total_step},"
                    f"reward = {episode_reward},"
                    f"step = {episode_step}"
                )
                return final_observation
            else:
                return next_state

        # Determine the real next state
        real_next_state = next_state

        if episode_finish.any():
            real_next_state = np.array(
                list(
                    map(
                        process_episode_finished_environment,
                        range(self.num_envs),
                        episode_finish,
                        self.episode_reward,
                        self.episode_step,
                        self.total_step,
                        info["final_observation"],
                        next_state,
                    )
                )
            )

        return (
            self.episode_step,
            state,
            real_next_state,
            action,
            reward,
            self.terminated,
            self.truncated,
            info,
        )

    def get_statistics(self) -> dict:
        """
        Get the statistics about the experiences generated so far.

        Returns:
            dict: The statistics as a dictionary. Contains the average reward sum and average step.

        Raises:
            Warning: If this method is called but calc_stats was set to False during initialization.
        """
        if self.calc_stats:
            stats = {
                "average_reward_sum": mean_or_nan(self.reward_sum_record),
                "average_step": mean_or_nan(self.step_record),
            }
            clear_if_maxlen_is_none(self.reward_sum_record, self.step_record)
            return stats
        else:
            self.logger.warning("get_statistics() is called even though the calc_stats is False.")
