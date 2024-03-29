import argparse

import wandb
from rl_algos.agents import ATRPO
from rl_algos.experiments import Evaluator, Recoder, training
from rl_algos.modules import ZScoreFilter
from rl_algos.utils import logger, manual_seed
from rl_algos.wrappers import make_env, register_reset_env, vectorize_env


def train_atrpo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Hopper-v4")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--lambd", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--vf_epoch", type=int, default=5)
    parser.add_argument("--vf_batch_size", type=int, default=64)
    parser.add_argument("--conjugate_gradient_damping", type=float, default=1e-1)
    parser.add_argument("--use_state_normalizer", action="store_true")
    parser.add_argument("--max_step", type=int, default=10**6)
    parser.add_argument("--eval_interval", type=int, default=5 * 10**4)
    parser.add_argument("--num_evaluate", type=int, default=10)
    parser.add_argument("--num_videos", type=int, default=3)
    args = parser.parse_args()

    wandb.init(project="trpo", tags=["atrpo", args.env_id], config=args)

    register_reset_env()

    manual_seed(args.seed)

    env = vectorize_env(env_id="reset_env/" + args.env_id, num_envs=args.num_envs)

    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space.shape[-1]

    logger.info(f"env = {env}")
    logger.info(f"dim_state = {dim_state}")
    logger.info(f"dim_action = {dim_action}")
    logger.info(f"action_space = {env.action_space}")
    logger.info(f"max_episode_steps = {env.spec.max_episode_steps}")

    agent = ATRPO(
        dim_state=dim_state,
        dim_action=dim_action,
        lambd=args.lambd,
        entropy_coef=0.0,
        vf_epoch=args.vf_epoch,
        vf_batch_size=args.vf_batch_size,
        vf_optimizer_kwargs={"lr": args.lr} if args.lr is not None else {},
        state_normalizer=ZScoreFilter(dim_state) if args.use_state_normalizer else None,
        conjugate_gradient_damping=args.conjugate_gradient_damping,
        update_interval=args.num_envs * 1000
        if args.update_interval is None
        else args.update_interval,
    )

    evaluator = Evaluator(
        env=make_env(args.env_id),
        eval_interval=args.eval_interval,
        num_evaluate=args.num_evaluate,
    )

    recoder = (
        Recoder(
            env=make_env(args.env_id, render_mode="rgb_array"),
            record_interval=args.max_step // args.num_videos,
        )
        if args.num_videos > 0
        else None
    )

    agent = training(
        env=env,
        agent=agent,
        max_steps=args.max_step,
        recorder=recoder,
        evaluator=evaluator,
    )


if __name__ == "__main__":
    train_atrpo()
