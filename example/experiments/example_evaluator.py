import logging
from rlrl.experiments import Evaluator
from statistics import mean, stdev


def main():
    import gym

    logging.basicConfig(level=logging.INFO)
    env = gym.make("Swimmer-v3")

    def actor(state):
        return env.action_space.sample()

    evaluator = Evaluator(env, 100)

    scores = evaluator.evaluate(actor)

    print(f"mean = {mean(scores)}, stdev = {stdev(scores)}")


if __name__ == "__main__":
    main()
