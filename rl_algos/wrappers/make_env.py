from functools import partial

import gymnasium
import numpy as np
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.vector_env import VectorEnv

from rl_algos.wrappers.cast_observation_reward import CastObservationToFloat32, CastRewardToFloat
from rl_algos.wrappers.normalize_action_space import NormalizeActionSpace


def make_env(env_id, **kwargs) -> gymnasium.Env:
    env = gymnasium.make(env_id, **kwargs)
    env = CastObservationToFloat32(env)
    env = CastRewardToFloat(env)
    env = NormalizeActionSpace(env)
    return env


def vectorize_env(env_id: str, num_envs: int = 1, env_fn=make_env) -> VectorEnv:
    env_fns = [partial(env_fn, env_id=env_id) for _ in range(num_envs)]
    if num_envs == 1:
        envs = SyncVectorEnv(env_fns)
        envs._rewards = envs._rewards.astype(np.float32)
    else:
        envs = AsyncVectorEnv(env_fns)

    dummy_env = env_fns[0]()

    if hasattr(dummy_env, "spec"):
        setattr(envs, "spec", dummy_env.spec)

    return envs
