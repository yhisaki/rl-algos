import gym
from gym.vector.async_vector_env import AsyncVectorEnv


def make_env(env_name):
    def _make():
        _env = gym.make(env_name)
        return _env

    return _make


def main():
    env_id = "Ant-v3"
    num_envs = 5
    vec_env = AsyncVectorEnv([make_env(env_id) for i in range(num_envs)])

    state = vec_env.reset()

    for i in range(5000):
        action = vec_env.action_space.sample()
        state, reward, done, _ = vec_env.step(action)
        if any(done):
            done_idx = [i for i, e in enumerate(done) if e]
            print(f"{done_idx}")


if __name__ == "__main__":
    main()
