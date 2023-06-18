import argparse

import wandb
from rl_algos.agents import DDPG
from rl_algos.experiments import Evaluator, Recoder, training
from rl_algos.utils import logger, manual_seed
from rl_algos.wrappers import make_env, vectorize_env


def train_ddpg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="InvertedPendulum-v4", type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--max_step", default=10**6, type=int)
    parser.add_argument("--eval_interval", type=int, default=10**4)
    parser.add_argument("--num_evaluate", type=int, default=10)
    parser.add_argument("--logging_interval", type=int, default=10**3)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--replay_start_size", default=10**4, type=int)
    parser.add_argument("--num_videos", type=int, default=3)
    args = parser.parse_args()

    wandb.init(project="rl_algos_example", name="ddpg", tags=[args.env_id])

    wandb.config.update(args)

    # fix seed
    manual_seed(args.seed)

    # make environment
    env = vectorize_env(env_id=args.env_id, num_envs=args.num_envs)

    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space.shape[-1]

    logger.info(f"env = {env}")
    logger.info(f"dim_state = {dim_state}")
    logger.info(f"dim_action = {dim_action}")
    logger.info(f"action_space = {env.action_space}")
    logger.info(f"max_episode_steps = {env.spec.max_episode_steps}")

    # make agent
    agent = DDPG(
        dim_state=dim_state,
        dim_action=dim_action,
        gamma=args.gamma,
        replay_start_size=args.replay_start_size,
    )

    evaluator = Evaluator(
        env=make_env(args.env_id),
        eval_interval=args.eval_interval,
        num_evaluate=args.num_evaluate,
    )
    recoder = (
        Recoder(
            env=make_env(args.env_id),
            record_interval=args.max_step // args.num_videos,
        )
        if args.num_videos > 0
        else None
    )

    agent = training(
        env=env,
        agent=agent,
        max_steps=args.max_step,
        logging_interval=args.logging_interval,
        recorder=recoder,
        evaluator=evaluator,
    )


if __name__ == "__main__":
    train_ddpg()
