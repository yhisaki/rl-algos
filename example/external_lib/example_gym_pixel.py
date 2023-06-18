import contextlib

import gymnasium
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper


def make_env(
    env_id: str,
    num_envs: int = 1,
):
    def _make():
        _env = gymnasium.make(env_id, render_mode="rgb_array")
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
def gymnasium_wrapping(env, wrapper):
    try:
        env = wrapper(env)
    finally:
        env = env.env


def main():
    # wandb.init(project="example_rl_algos")
    env = gymnasium.make("Hopper-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(env, pixels_only=False)
    env.reset()
    pixels = []
    while True:
        (state, reward, terminated, truncated, info) = env.step(env.action_space.sample())
        pixels.append(state["pixels"].transpose(2, 0, 1))
        if terminated or truncated:
            # wandb.log({"video": wandb.Video(np.array(pixels), fps=60)})
            break


if __name__ == "__main__":
    main()
