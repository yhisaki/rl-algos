# from typing import Optional

from typing import Optional
import gym
from gym.vector import AsyncVectorEnv

from rlrl.wrappers.cast_observation_reward import CastObservationToFloat32, CastRewardToFloat32
from rlrl.wrappers.normalize_action_space import NormalizeActionSpace

# from rlrl.wrappers.video_record import NumpyArrayMonitor
from rlrl.wrappers.single_as_vector_env import SingleAsVectorEnv

# def make_env(env_id: str, seed: Optional[int] = None, monitor: bool = False, monitor_args={}):
#     env = gym.make(env_id)
#     env = CastObservationToFloat32(env)
#     env = CastRewardToFloat32(env)
#     env = NormalizeActionSpace(env)
#     if seed is not None:
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#     if monitor:
#         env = NumpyArrayMonitor(env, **monitor_args)
#     return env


def make_env(env_id, seed: Optional[int] = None):
    env = gym.make(env_id)
    env = CastObservationToFloat32(env)
    env = CastRewardToFloat32(env)
    env = NormalizeActionSpace(env)
    if seed is not None:
        env.seed(seed)
    return env


def make_envs_for_training(env_id: str, num_envs: int, seeds):
    def _make_env():
        env = gym.make(env_id)
        env = CastObservationToFloat32(env)
        env = CastRewardToFloat32(env)
        env = NormalizeActionSpace(env)
        return env

    if num_envs == 1:
        envs = SingleAsVectorEnv(_make_env())
    else:
        envs = AsyncVectorEnv([_make_env for _ in range(num_envs)])

    dummy_env = _make_env()

    if hasattr(dummy_env, "spec"):
        setattr(envs, "spec", dummy_env.spec)

    if seeds is not None:
        envs.seed(seeds)

    return envs
