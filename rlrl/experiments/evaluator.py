import logging

# import multiprocessing as mp
from typing import List

from gym import Env


# TODO implement multiprocessing
class Evaluator(object):
    def __init__(self, env: Env, num_evaluate: int, logger=logging.getLogger(__name__)) -> None:
        super().__init__()
        # ctx = mp.get_context()
        self.env = env
        self.num_evaluate = num_evaluate
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
