import argparse
import logging
from statistics import mean, stdev

import wandb
from rlrl.agents.atrpo_agent import AtrpoAgent
from rlrl.experiments import Evaluator, GymMDP
from rlrl.nn.z_score_filter import ZScoreFilter  # NOQA
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.wrappers import make_env, make_envs_for_training


def _add_header_to_dict_key(d: dict, header: str):
    return {header + "/" + k: v for k, v in d.items()}


def train_atrpo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Swimmer-v2")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--lambd", type=float, default=0.97)
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()

    wandb.init(project="trpo", tags=["atrpo", args.env_id])

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

    agent = AtrpoAgent(
        dim_state=dim_state,
        dim_action=dim_action,
        entropy_coef=0.0,
        vf_epoch=5,
        lambd=args.lambd,
        conjugate_gradient_damping=1e-1,
        update_interval=args.num_envs * 1000
        if args.update_interval is None
        else args.update_interval,
    )

    evaluator = Evaluator(
        env=make_env(args.env_id, args.seed), eval_interval=50000, num_evaluate=100
    )

    def actor(state):
        return agent.act(state)

    interactions = GymMDP(env, actor, max_step=5e5)

    wandb.config.num_envs = args.num_envs
    wandb.config.seed = args.seed
    wandb.config.lambd = agent.lambd
    wandb.config.update_interval = agent.update_interval

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
            scores = evaluator.evaluate_if_necessary(interactions.total_step, actor)
            if scores is not None:
                print(f"Evaluate Agent: mean_score: {mean(scores)} (stdev: {stdev(scores)})")
                wandb.log(
                    {
                        "step": interactions.total_step.sum(),
                        "eval/mean": mean(scores),
                        "eval/stdev": stdev(scores),
                    }
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

    with agent.eval_mode():
        videos = evaluator.record_videos(actor, num_videos=1, pixel=True)
        for video in videos:
            wandb.log({"video": wandb.Video(video, fps=60)})


if __name__ == "__main__":
    train_atrpo()
