import contextlib

import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.wrappers.pixel_observation import PixelObservationWrapper


def make_env(
    env_id: str,
    num_envs: int = 1,
):
    def _make():
        _env = gym.make(env_id)
        return _env

    if num_envs == 1:
        env = SyncVectorEnv([_make])
    if num_envs > 1:
        env = AsyncVectorEnv([_make for _ in range(num_envs)])
        dummy_env = _make()
        setattr(env, "spec", dummy_env.spec)

        del dummy_env
    else:
        env = _make()
    return env


@contextlib.contextmanager
def gym_wrapping(env, wrapper):
    try:
        env = wrapper(env)
    finally:
        env = env.env


def main():
    # wandb.init(project="example_rlrl")
    env = gym.make("Hopper-v2")
    env = PixelObservationWrapper(env, pixels_only=False)
    env.reset()
    pixels = []
    while True:
        (state, reward, done, info) = env.step(env.action_space.sample())
        pixels.append(state["pixels"].transpose(2, 0, 1))
        if done:
            # wandb.log({"video": wandb.Video(np.array(pixels), fps=60)})
            break


if __name__ == "__main__":
    main()
