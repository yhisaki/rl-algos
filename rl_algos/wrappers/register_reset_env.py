from functools import partial

from gymnasium.envs.registration import register

from rl_algos.wrappers.reset_cost_wrapper import ResetCostWrapper


def _create_mujoco_envs(env_id: str, **kwargs):
    from gymnasium.envs.mujoco.ant_v4 import AntEnv
    from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
    from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
    from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
    from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
    from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv

    env_id_to_env = {
        "Ant-v4": AntEnv,
        "HalfCheetah-v4": HalfCheetahEnv,
        "Hopper-v4": HopperEnv,
        "Humanoid-v4": HumanoidEnv,
        "Swimmer-v4": SwimmerEnv,
        "Walker2d-v4": Walker2dEnv,
    }

    if kwargs.get("reset_cost") is not None:
        return ResetCostWrapper(env_id_to_env[env_id](), reset_cost=kwargs["reset_cost"])
    else:
        return ResetCostWrapper(env_id_to_env[env_id]())


def register_reset_env():
    register(
        id="reset_env/HalfCheetah-v4",
        entry_point=partial(_create_mujoco_envs, env_id="HalfCheetah-v4"),
        max_episode_steps=1000,
        reward_threshold=None,
    )
    register(
        id="reset_env/Hopper-v4",
        entry_point=partial(_create_mujoco_envs, env_id="Hopper-v4"),
        max_episode_steps=1000,
        reward_threshold=None,
    )
    register(
        id="reset_env/Swimmer-v4",
        entry_point=partial(_create_mujoco_envs, env_id="Swimmer-v4"),
        max_episode_steps=1000,
        reward_threshold=None,
    )
    register(
        id="reset_env/Ant-v4",
        entry_point=partial(_create_mujoco_envs, env_id="Ant-v4"),
        max_episode_steps=1000,
        reward_threshold=None,
    )
    register(
        id="reset_env/Walker2d-v4",
        entry_point=partial(_create_mujoco_envs, env_id="Walker2d-v4"),
        max_episode_steps=1000,
        reward_threshold=None,
    )
    register(
        id="reset_env/Humanoid-v4",
        entry_point=partial(_create_mujoco_envs, env_id="Humanoid-v4"),
        max_episode_steps=1000,
        reward_threshold=None,
    )


if __name__ == "__main__":
    import gymnasium

    register_reset_env()
    env = gymnasium.make("reset_env/Hopper-v4")
    print(env)
