import argparse
import logging
from statistics import mean, stdev

import wandb
from rlrl.agents import TrpoAgent
from rlrl.experiments import Evaluator, GymMDP
from rlrl.nn.z_score_filter import ZScoreFilter  # NOQA
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.wrappers import make_env, make_envs_for_training


def _add_header_to_dict_key(d: dict, header: str):
    return {header + "/" + k: v for k, v in d.items()}


def train_trpo():
    parser = argparse.ArgumentParser()

    wandb.init(project="trpo")

    parser.add_argument("--env_id", type=str, default="Hopper-v2")
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()

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
        gamma=0.995,
        entropy_coef=0.0,
        vf_epoch=5,
        conjugate_gradient_damping=1e-1,
        update_interval=args.num_envs * 1000,
    )

    evaluator = Evaluator(env=make_env(args.env_id), num_evaluate=100)

    def actor(state):
        return agent.act(state)

    interactions = GymMDP(env, actor, max_step=5e5)

    for steps, states, next_states, actions, rewards, dones in interactions:
        agent.observe(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            terminals=is_state_terminal(env, steps, dones),
            resets=dones,
        )
        if interactions.total_step.sum() % 50000 < interactions.num_envs:
            with agent.eval_mode():
                scores = evaluator.evaluate(actor)
                print(
                    "\033[31m Evaluate Agent \033[0m"
                    f"mean_score = {mean(scores)} (stdev = {stdev(scores)})"
                )
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


if __name__ == "__main__":
    train_trpo()
