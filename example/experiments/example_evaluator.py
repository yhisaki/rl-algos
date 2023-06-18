import logging
from statistics import mean, stdev

from rl_algos.experiments import Evaluator


def main():
    import gymnasium

    logging.basicConfig(level=logging.INFO)
    env = gymnasium.make("Swimmer-v4")

    def actor(state):
        return env.action_space.sample()

    evaluator = Evaluator(env, 100)

    scores = evaluator.evaluate(actor)

    print(f"mean = {mean(scores)}, stdev = {stdev(scores)}")


if __name__ == "__main__":
    main()
