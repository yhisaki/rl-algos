import argparse
import logging

import wandb
from rl_algos.agents.td3_agent import TD3
from rl_algos.experiments import Evaluator, Recoder, training
from rl_algos.utils import manual_seed
from rl_algos.wrappers import make_env, vectorize_env


def train_td3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--policy_update_delay", type=int, default=2)
    parser.add_argument("--max_step", type=int, default=10**6)
    parser.add_argument("--eval_interval", type=int, default=10**4)
    parser.add_argument("--logging_interval", type=int, default=10**3)
    parser.add_argument("--num_evaluate", type=int, default=10)
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()

    wandb.init(project="td3", tags=["td3", args.env_id], config=args)

    wandb.config.update(args)

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(__name__)

    manual_seed(args.seed)

    env = vectorize_env(env_id=args.env_id, num_envs=args.num_envs)
    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space.shape[-1]

    logger.info(f"env = {env}")
    logger.info(f"dim_state = {dim_state}")
    logger.info(f"dim_action = {dim_action}")
    logger.info(f"action_space = {env.action_space}")
    logger.info(f"max_episode_steps = {env.spec.max_episode_steps}")

    agent = TD3(
        dim_state=dim_state,
        dim_action=dim_action,
        gamma=args.gamma,
        batch_size=args.batch_size,
        policy_update_delay=args.policy_update_delay,
        policy_optimizer_kwargs={"lr": args.lr},
        q_optimizer_kwargs={"lr": args.lr},
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
    train_td3()
