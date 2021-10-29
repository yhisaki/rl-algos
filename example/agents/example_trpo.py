import argparse
import logging
from statistics import mean, stdev

import wandb
from rlrl.agents import TrpoAgent
from rlrl.experiments import Evaluator, GymMDP
from rlrl.modules import ZScoreFilter
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.wrappers import make_env, make_envs_for_training


def _add_header_to_dict_key(d: dict, header: str):
    return {header + "/" + k: v for k, v in d.items()}


def train_trpo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Hopper-v2")
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
    parser.add_argument("--max_step", type=int, default=5e5)
    parser.add_argument("--eval_interval", type=int, default=5e4)
    parser.add_argument("--num_evaluate", type=int, default=10)
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()

    wandb.init(project="trpo", tags=["trpo", args.env_id], config=args)

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(__name__)

    manual_seed(args.seed)

    env = make_envs_for_training(args.env_id, args.num_envs, args.seed)
    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space[0].shape[-1]

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
        record_interval=args.max_step // args.num_videos,
    )

    def actor(state):
        return agent.act(state)

    interactions = GymMDP(env, actor, max_step=args.max_step)

    for steps, states, next_states, actions, rewards, dones in interactions:
        agent.observe(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            terminals=is_state_terminal(env, steps, dones),
            resets=dones,
        )
        with agent.eval_mode():

            # Evaluate
            scores = evaluator.evaluate_if_necessary(interactions.total_step, actor)
            if len(scores) != 0:
                print(f"Evaluate Agent: mean_score: {mean(scores)} (stdev: {stdev(scores)})")
                wandb.log(
                    {
                        "step": interactions.total_step.sum(),
                        "eval/mean": mean(scores),
                        "eval/stdev": stdev(scores),
                    }
                )

            # Record videos
            videos = evaluator.record_videos_if_necessary(
                interactions.total_step, actor, pixel=True
            )
            for video in videos:
                wandb.log(
                    {"step": interactions.total_step.sum(), "video": wandb.Video(video, fps=60)}
                )

        if agent.just_updated:
            agent_stats = agent.get_statistics()
            gym_stats = interactions.get_statistics()
            wandb.log(
                {
                    "step": interactions.total_step.sum(),
                    **_add_header_to_dict_key(agent_stats, "train"),
                    **_add_header_to_dict_key(gym_stats, "train"),
                }
            )


if __name__ == "__main__":
    train_trpo()
