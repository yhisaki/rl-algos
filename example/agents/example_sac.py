import argparse
import logging
from statistics import mean, stdev

import wandb
from rlrl.agents import SacAgent
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.experiments import Evaluator, GymMDP
from rlrl.wrappers import make_envs_for_training, make_env


def _add_header_to_dict_key(d: dict, header: str):
    return {header + "/" + k: v for k, v in d.items()}


def train_sac():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="HalfCheetah-v3", type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--max_step", default=10 ** 6, type=int)
    parser.add_argument("--eval_interval", type=int, default=10 ** 4)
    parser.add_argument("--num_evaluate", type=int, default=10)
    parser.add_argument("--agent_logging_interval", type=int, default=10 ** 3)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--t_init", default=10 ** 4, type=int)
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    wandb.init(project="rlrl_example", name="soft_actor_critic", tags=[args.env_id])

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(__name__)

    # fix seed
    manual_seed(args.seed)

    # make environment
    env = make_envs_for_training(
        args.env_id,
        args.num_envs,
        args.seed,
    )

    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space[0].shape[-1]

    logger.info(f"env = {env}")
    logger.info(f"dim_state = {dim_state}")
    logger.info(f"dim_action = {dim_action}")
    logger.info(f"action_space = {env.action_space}")
    logger.info(f"max_episode_steps = {env.spec.max_episode_steps}")

    # make agent
    agent = SacAgent(
        dim_state=dim_state,
        dim_action=dim_action,
        gamma=args.gamma,
        num_random_act=args.t_init,
    )

    evaluator = Evaluator(
        env=make_env(args.env_id, args.seed),
        eval_interval=args.eval_interval,
        num_evaluate=args.num_evaluate,
        record_interval=args.max_step // args.num_videos,
    )

    wandb.config.update(args)

    def actor(state):
        return agent.act(state)

    interactions = GymMDP(
        env,
        actor,
        max_step=args.max_step,
    )
    for step, states, next_states, actions, rewards, dones in interactions:
        agent.observe(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            terminals=is_state_terminal(env, step, dones),
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

        if agent.just_updated and (interactions.total_step % args.agent_logging_interval == 0):
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
    train_sac()
