import argparse
import logging

import wandb
from rlrl.agents import TrpoAgent
from rlrl.experiments import Evaluator, Recoder, training
from rlrl.modules import ZScoreFilter
from rlrl.utils import manual_seed
from rlrl.wrappers import make_env, vectorize_env


def train_trpo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Hopper-v3")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lambd", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vf_epoch", type=int, default=5)
    parser.add_argument("--vf_batch_size", type=int, default=64)
    parser.add_argument("--conjugate_gradient_damping", type=float, default=1e-1)
    parser.add_argument("--use_state_normalizer", action="store_true")
    parser.add_argument("--max_step", type=int, default=10 ** 6)
    parser.add_argument("--eval_interval", type=int, default=5 * 10 ** 4)
    parser.add_argument("--num_evaluate", type=int, default=10)
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()

    wandb.init(project="trpo", tags=["trpo", args.env_id], config=args)

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(__name__)

    manual_seed(args.seed)

    env = vectorize_env(env_id=args.env_id, num_envs=args.num_envs, seed=args.seed)
    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space.shape[-1]

    logger.info(f"env = {env}")
    logger.info(f"dim_state = {dim_state}")
    logger.info(f"dim_action = {dim_action}")
    logger.info(f"action_space = {env.action_space}")
    logger.info(f"max_episode_steps = {env.spec.max_episode_steps}")

    agent = TrpoAgent(
        dim_state,
        dim_action,
        gamma=args.gamma,
        lambd=args.lambd,
        entropy_coef=0.0,
        vf_epoch=args.vf_epoch,
        vf_batch_size=args.vf_batch_size,
        vf_optimizer_kwargs={"lr": args.lr},
        state_normalizer=ZScoreFilter(dim_state) if args.use_state_normalizer else None,
        conjugate_gradient_damping=args.conjugate_gradient_damping,
        update_interval=args.num_envs * 1000
        if args.update_interval is None
        else args.update_interval,
    )

    evaluator = Evaluator(
        env=make_env(args.env_id, args.seed),
        eval_interval=args.eval_interval,
        num_evaluate=args.num_evaluate,
    )

    recoder = (
        Recoder(
            env=make_env(args.env_id, args.seed),
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
    train_trpo()
