import argparse
import logging
from typing import Optional
import gym

# import wandb
from rlrl.utils import manual_seed
from rlrl.experiments import GymMDP
from rlrl.agents.trpo_agent import TrpoAgent
from rlrl.nn.z_score_filter import ZScoreFilter  # NOQA
from rlrl.utils import is_state_terminal
from rlrl.wrappers import (
    CastObservationToFloat32,
    CastRewardToFloat32,
)


def make_env(
    env_id: str,
    num_envs: int = 1,
    seed: Optional[int] = None,
):
    def _make():
        _env = gym.make(env_id)
        _env = CastObservationToFloat32(_env)
        _env = CastRewardToFloat32(_env)
        return _env

    if num_envs > 1:
        env = gym.vector.async_vector_env.AsyncVectorEnv([_make for _ in range(num_envs)])
        dummy_env = _make()
        setattr(env, "spec", dummy_env.spec)

        del dummy_env
    else:
        env = _make()
    return env


def train_trpo():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    # parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--env_id", type=str, default="Hopper-v2")
    parser.add_argument("--num_envs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    args = parser.parse_args()

    manual_seed(args.seed)

    env = make_env(args.env_id, args.num_envs, args.seed)
    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space.shape[-1] if args.num_envs == 1 else env.action_space[0].shape[-1]

    logger.info(f"env = {env}")
    logger.info(f"dim_state = {dim_state}")
    logger.info(f"dim_action = {dim_action}")
    logger.info(f"action_space = {env.action_space}")
    logger.info(f"max_episode_steps = {env.spec.max_episode_steps}")

    agent = TrpoAgent(
        dim_state,
        dim_action,
        gamma=0.995,
        entropy_coef=0.0,
        vf_epoch=5,
        conjugate_gradient_damping=1e-1,
    )

    def actor(state):
        return agent.act(state)

    interactions = GymMDP(
        env,
        actor,
        max_step=1e7,
    )

    pre_update_step = 0
    for step, state, next_state, action, reward, done in interactions:
        agent.observe(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            terminal=is_state_terminal(env, step, done),
            reset=done,
        )

        if (interactions.total_step.sum() - pre_update_step) >= 5000:
            agent.update()
            pre_update_step = interactions.total_step.sum()


if __name__ == "__main__":
    train_trpo()
