import gymnasium
from gymnasium.vector.async_vector_env import AsyncVectorEnv

from rl_algos.experiments import TransitionGenerator


def example_single_env(env_id: str):
    env = gymnasium.make(env_id)

    def actor(_):
        return [env.action_space.sample()]

    interactions = TransitionGenerator(env, actor, max_step=1000)

    for _ in interactions:
        pass


def example_vector_env(env_id: str):
    def _make():
        return gymnasium.make(env_id)

    env = AsyncVectorEnv([_make for _ in range(4)])
    print(env.observation_space)

    def actor(_):
        return env.action_space.sample()

    interactions = TransitionGenerator(env, actor, max_episode=3)

    for _ in interactions:
        pass


if __name__ == "__main__":
    # example_single_env("Humanoid-v4")
    example_vector_env("Humanoid-v4")
