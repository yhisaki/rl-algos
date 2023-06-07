import logging

# import multiprocessing as mp
from typing import List, Optional

from gymnasium import Env


# TODO implement multiprocessing
class Evaluator(object):
    def __init__(
        self,
        env: Env,
        num_evaluate: int,
        eval_interval: int = 5e4,
        logger=logging.getLogger(__name__),
    ) -> None:
        super().__init__()
        # ctx = mp.get_context()
        self.env = env

        self.num_evaluate = num_evaluate  # Number of episodes used per evaluation.

        self.eval_interval = eval_interval
        self.pre_eval_step = 0

        self.logger = logger

    def evaluate(self, actor) -> List[float]:
        def _evaluate_once(i):
            state, _ = self.env.reset()
            reward_sum = 0
            step = 0
            while True:
                action = actor(state)
                state, reward, terminated, truncated, _ = self.env.step(action=action)
                step += 1
                reward_sum += reward
                if terminated or truncated:
                    self.logger.info(
                        f"Evaluate Actor {i+1}/{self.num_evaluate}, "
                        f"Reward Sum: {reward_sum}, Step: {step}"
                    )
                    return reward_sum

        scores = [_evaluate_once(i) for i in range(self.num_evaluate)]
        return scores

    def evaluate_if_necessary(self, total_steps: int, actor) -> Optional[List[float]]:
        scores = []
        n = total_steps // self.eval_interval
        if (self.pre_eval_step < n * self.eval_interval) and (
            n * self.eval_interval <= total_steps
        ):
            scores = self.evaluate(actor)
        self.pre_eval_step = total_steps
        return scores
