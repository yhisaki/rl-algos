import argparse
import os
import logging

import wandb
from rl_algos.agents.research import ATD3
from rl_algos.experiments import Evaluator, Recoder, training
from rl_algos.utils import manual_seed
from rl_algos.wrappers import ResetCostWrapper, make_env, vectorize_env


def train_atd3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Ant-v4")
    parser.add_argument("--project", default="average-reward-rl", type=str)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_step", type=int, default=10**6)
    parser.add_argument("--reset_cost", default="auto")
    parser.add_argument("--target_terminal_probability", type=float, default=1 / 1000)
    parser.add_argument("--eval_interval", type=int, default=10**4)
    parser.add_argument("--logging_interval", type=int, default=10**3)
    parser.add_argument("--num_evaluate", type=int, default=10)
    parser.add_argument("--num_videos", type=int, default=0)
    parser.add_argument("--save_model", action="store_false")
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()

    wandb.init(project=args.project, tags=["atd3", args.env_id], config=args, group=args.group)

    wandb.config.update(args)

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(__name__)

    manual_seed(args.seed)

    def _make_env(*env_args, **env_kwargs):
        return ResetCostWrapper(make_env(*env_args, **env_kwargs), reset_cost=float("nan"))

    env = vectorize_env(env_id=args.env_id, num_envs=args.num_envs, env_fn=_make_env)

    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space.shape[-1]

    logger.info(f"env = {env}")
    logger.info(f"dim_state = {dim_state}")
    logger.info(f"dim_action = {dim_action}")
    logger.info(f"action_space = {env.action_space}")
    logger.info(f"max_episode_steps = {env.spec.max_episode_steps}")

    agent = ATD3(
        dim_state=dim_state,
        dim_action=dim_action,
        target_terminal_probability=args.target_terminal_probability,
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

    if args.save_model:
        os.mkdir(os.path.join(wandb.run.dir, "model"))
        agent.save(os.path.join(wandb.run.dir, "model"))


if __name__ == "__main__":
    train_atd3()
