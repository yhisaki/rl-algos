from rlrl.experiments import GymMDP
import gym
from gym.vector.async_vector_env import AsyncVectorEnv


def example_single_env():
    env = gym.make("Swimmer-v2")

    def actor(state):
        return [env.action_space.sample()]

    interactions = GymMDP(env, actor, max_episode=2)

    for epi_step, state, next_state, action, reward, done in interactions:
        print(state)


def example_vector_env():
    def _make():
        return gym.make("Hopper-v2")

    env = AsyncVectorEnv([_make for _ in range(3)])
    print(env.observation_space)

    def actor(state):
        return env.action_space.sample()

    interactions = GymMDP(env, actor, max_episode=2)

    for epi_step, state, next_state, action, reward, done in interactions:
        for es in state:
            print(es)
        break


if __name__ == "__main__":
    example_single_env()
