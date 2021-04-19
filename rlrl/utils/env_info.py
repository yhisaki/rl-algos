from collections import namedtuple

EnvInfo = namedtuple(
    "EnvInfo",
    ("dim_state", "dim_action", "action_high", "action_low", "max_episode_steps")
)


def get_env_info(env):
  env_info = EnvInfo(
      dim_state=env.observation_space.shape[0],
      dim_action=env.action_space.shape[0],
      action_high=env.action_space.high,
      action_low=env.action_space.low,
      max_episode_steps=env._max_episode_steps
  )
  return env_info
