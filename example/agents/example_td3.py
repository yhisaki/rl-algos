import argparse
import logging

import wandb
from rlrl.agents.td3_agent import Td3Agent
from rlrl.experiments import Evaluator, GymMDP
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.wrappers import make_env, make_envs_for_training


def train_td3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v2")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_step", type=int, default=5e5)
    parser.add_argument("--eval_interval", type=int, default=10 ** 4)
    parser.add_argument("--num_evaluate", type=int, default=10)
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()

    wandb.init(project="td3", tags=["td3", args.env_id], config=args)

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

    agent = Td3Agent(dim_state=dim_state, dim_action=dim_action, gamma=args.gamma)

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
        )
        with agent.eval_mode():
            evaluator.evaluate_if_necessary(interactions.total_step, actor)


if __name__ == "__main__":
    train_td3()
