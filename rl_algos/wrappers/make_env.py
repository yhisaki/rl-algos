from functools import partial

import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.vector_env import VectorEnv

from rl_algos.wrappers.cast_observation_reward import CastObservationToFloat32, CastRewardToFloat32
from rl_algos.wrappers.normalize_action_space import NormalizeActionSpace
from rl_algos.wrappers.single_as_vector_env import SingleAsVectorEnv


def make_env(env_id, seed: int = 0) -> gym.Env:
    env = gym.make(env_id)
    env = CastObservationToFloat32(env)
    env = CastRewardToFloat32(env)
    env = NormalizeActionSpace(env)
    env.seed(seed)
    return env


def vectorize_env(env_id: str, num_envs: int = 1, env_fn=make_env, seed=0) -> VectorEnv:
    env_fns = [partial(env_fn, env_id=env_id) for _ in range(num_envs)]
    if num_envs == 1:
        envs = SingleAsVectorEnv(env_fns[0]())
    else:
        envs = AsyncVectorEnv(env_fns)

    dummy_env = env_fns[0]()

    if hasattr(dummy_env, "spec"):
        setattr(envs, "spec", dummy_env.spec)

    envs.seed(seed)

    return envs
