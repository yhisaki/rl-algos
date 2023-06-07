import logging
from statistics import mean, stdev
from typing import Optional

from gymnasium import Env

import wandb
from rl_algos.agents.agent_base import AgentBase
from rl_algos.experiments.evaluator import Evaluator
from rl_algos.experiments.recorder import Recoder
from rl_algos.experiments.transition_generator import TransitionGenerator
from rl_algos.utils import is_state_terminal


def get_average_reward(env: Env, actor, reset_cost: float):
    T = 10000
    state = env.reset()
    reward_sum = 0
    for i in range(T):
        action = actor(state)
        next_state, reward, done, info = env.step(action=action)
        reward_sum += reward
        if done:
            state = env.reset()
            reward_sum -= reset_cost
        else:
            state = next_state

    return reward_sum / T


def __add_header_to_dict_key(d: dict, header: str):
    return {header + "/" + k: v for k, v in d.items()}


def training_average_rl(
    env: Env,
    env_for_average_reward: Env,
    agent: AgentBase,
    max_steps: int,
    logging_interval: int = 1,
    evaluator: Optional[Evaluator] = None,
    recorder: Optional[Recoder] = None,
    logger: logging.Logger = logging.getLogger(__name__),
):
    def actor(state):
        return agent.act(state)

    interactions = TransitionGenerator(env, actor, max_step=max_steps)

    for steps, states, next_states, actions, rewards, terminal, truncated, info in interactions:
        agent.observe(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            terminals=terminal,
            resets=terminal | truncated,
        )
        with agent.eval_mode():
            # Evaluate
            if evaluator is not None:
                scores = evaluator.evaluate_if_necessary(interactions.total_step.sum(), actor)
                if len(scores) > 0:
                    average_reward = get_average_reward(
                        env=env_for_average_reward, actor=actor, reset_cost=agent.reset_cost
                    )
                    logger.info(
                        f"Evaluate Agent: mean_score: {mean(scores)} (stdev: {stdev(scores)})"
                    )
                    wandb.log(
                        {
                            "step": interactions.total_step.sum(),
                            "eval/mean": mean(scores),
                            "eval/stdev": stdev(scores),
                            "average_reward": average_reward,
                        }
                    )
            if recorder is not None:
                # Record videos
                videos = recorder.record_videos_if_necessary(interactions.total_step.sum(), actor)
                for video in videos:
                    wandb.log(
                        {
                            "step": interactions.total_step.sum(),
                            "video": wandb.Video(video, fps=60, format="mp4"),
                        }
                    )

        if agent.just_updated and (interactions.total_step.sum() % logging_interval == 0):
            stats = agent.get_statistics()
            logger.info(stats)
            wandb.log(
                {
                    "step": interactions.total_step.sum(),
                    **__add_header_to_dict_key(stats, "train"),
                    **__add_header_to_dict_key(interactions.get_statistics(), "train"),
                }
            )

    return agent
