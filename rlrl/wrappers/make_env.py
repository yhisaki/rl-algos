from typing import Optional

import gym

from rlrl.wrappers.cast_observation_reward import CastObservationToFloat32, CastRewardToFloat
from rlrl.wrappers.normalize_action_space import NormalizeActionSpace
from rlrl.wrappers.video_record import NumpyArrayMonitor


def make_env(env_id: str, seed: Optional[int] = None, monitor: bool = False, monitor_args={}):
    env = gym.make(env_id)
    env = NormalizeActionSpace(CastRewardToFloat(CastObservationToFloat32(env)))
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    if monitor:
        env = NumpyArrayMonitor(env, **monitor_args)
    return env
