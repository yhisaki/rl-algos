import logging

# import multiprocessing as mp
from typing import List, Optional

import numpy as np
from gym import Env

from rlrl.experiments.record_videos_from_actor import record_videos_from_actor


# TODO implement multiprocessing
class Evaluator(object):
    def __init__(
        self,
        env: Env,
        num_evaluate: int,
        eval_interval: int = 5e4,
        record_interval: int = 10e4,
        logger=logging.getLogger(__name__),
    ) -> None:
        super().__init__()
        # ctx = mp.get_context()
        self.env = env

        self.num_evaluate = num_evaluate  # Number of episodes used per evaluation.

        self.eval_interval = eval_interval
        self.pre_eval_step = 0

        self.record_interval = record_interval
        self.pre_record_step = 0

        self.logger = logger

    def evaluate(self, actor) -> List[float]:
        def _evaluate_once(i):
            state = self.env.reset()
            reward_sum = 0
            step = 0
            while True:
                action = actor(state)
                state, reward, done, _ = self.env.step(action=action)
                step += 1
                reward_sum += reward
                if done:
                    self.logger.info(
                        f"Evaluate Actor {i+1}/{self.num_evaluate}, "
                        f"Reward Sum: {reward_sum}, Step: {step}"
                    )
                    return reward_sum

        scores = [_evaluate_once(i) for i in range(self.num_evaluate)]
        return scores

    def evaluate_if_necessary(self, total_steps: np.ndarray, actor) -> Optional[List[float]]:
        scores = []
        n = total_steps.sum() // self.eval_interval
        if (self.pre_eval_step < n * self.eval_interval) and (
            n * self.eval_interval <= total_steps.sum()
        ):
            scores = self.evaluate(actor)
        self.pre_eval_step = total_steps.sum()
        return scores

    def record_videos(self, actor, num_videos=1, pixel: bool = False, dir: str = None):
        return record_videos_from_actor(self.env, actor, num_videos, pixel, dir, self.logger)

    def record_videos_if_necessary(
        self, total_steps: np.ndarray, actor, num_videos=1, pixel: bool = False, dir: str = None
    ):
        videos = []
        n = total_steps.sum() // self.record_interval
        if (self.pre_record_step < n * self.record_interval) and (
            n * self.record_interval <= total_steps.sum()
        ):
            videos = self.record_videos(actor, num_videos, pixel, dir)
        self.pre_record_step = total_steps.sum()
        return videos
