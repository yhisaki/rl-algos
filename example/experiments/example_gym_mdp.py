from rlrl.experiments import GymMDP
import gym
from gym.vector.async_vector_env import AsyncVectorEnv
import logging


def example_single_env(env_id: str):
    env = gym.make(env_id)

    def actor(_):
        return [env.action_space.sample()]

    interactions = GymMDP(env, actor, max_step=1000)

    for _ in interactions:
        pass


def example_vector_env(env_id: str):
    def _make():
        return gym.make(env_id)

    env = AsyncVectorEnv([_make for _ in range(3)])
    print(env.observation_space)

    def actor(_):
        return env.action_space.sample()

    interactions = GymMDP(env, actor, max_episode=2)

    for _ in interactions:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_single_env("Humanoid-v3")
