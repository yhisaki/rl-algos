import argparse
import logging
from statistics import mean, stdev


import wandb
from rlrl.agents.td3_agent import Td3Agent
from rlrl.experiments import Evaluator, Recoder, GymMDP
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.wrappers import make_env, make_envs_for_training


def _add_header_to_dict_key(d: dict, header: str):
    return {header + "/" + k: v for k, v in d.items()}


def train_td3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v3")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--policy_update_delay", type=int, default=2)
    parser.add_argument("--max_step", type=int, default=10 ** 6)
    parser.add_argument("--eval_interval", type=int, default=10 ** 4)
    parser.add_argument("--agent_logging_interval", type=int, default=10 ** 3)
    parser.add_argument("--num_evaluate", type=int, default=10)
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()

    wandb.init(project="td3", tags=["td3", args.env_id], config=args)

    wandb.config.update(args)

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

    agent = Td3Agent(
        dim_state=dim_state,
        dim_action=dim_action,
        gamma=args.gamma,
        batch_size=args.batch_size,
        policy_update_delay=args.policy_update_delay,
    )

    evaluator = Evaluator(
        env=make_env(args.env_id, args.seed),
        eval_interval=args.eval_interval,
        num_evaluate=args.num_evaluate,
    )

    recoder = Recoder(
        env=make_env(args.env_id, args.seed),
        record_interval=args.max_step // args.num_videos if args.num_videos > 0 else -1,
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
            scores = evaluator.evaluate_if_necessary(interactions.total_step.sum(), actor)
            if len(scores) > 0:
                print(f"Evaluate Agent: mean_score: {mean(scores)} (stdev: {stdev(scores)})")
                wandb.log(
                    {
                        "step": interactions.total_step.sum(),
                        "eval/mean": mean(scores),
                        "eval/stdev": stdev(scores),
                    }
                )

            # Record videos
            videos = recoder.record_videos_if_necessary(interactions.total_step.sum(), actor)
            for video in videos:
                wandb.log(
                    {
                        "step": interactions.total_step.sum(),
                        "video": wandb.Video(video, fps=60, format="mp4"),
                    }
                )

        if agent.just_updated:
            wandb.log(
                {
                    "step": interactions.total_step.sum(),
                    **_add_header_to_dict_key(agent.get_statistics(), "train"),
                    **_add_header_to_dict_key(interactions.get_statistics(), "train"),
                }
            )


if __name__ == "__main__":
    train_td3()
