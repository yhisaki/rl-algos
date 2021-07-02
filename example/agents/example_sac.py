import argparse
from pprint import pprint
from typing import Optional

import gym

import wandb
from rlrl.agents import SacAgent
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.wrappers import (
    CastObservationToFloat32,
    CastRewardToFloat,
    NormalizeActionSpace,
    NumpyArrayMonitor,
)


def make_env(env_id: str, seed: Optional[int] = None, monitor: bool = False, monitor_args={}):
    env = gym.make(env_id)
    env = NormalizeActionSpace(CastRewardToFloat(CastObservationToFloat32(env)))
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    if monitor:
        env = NumpyArrayMonitor(env, **monitor_args)
    return env


def train_sac():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="Swimmer-v2", type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--max_step", default=int(1e6), type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--t_init", default=int(1e4), type=int)
    parser.add_argument("--save_video_interval", default=None, type=int)
    arg = parser.parse_args()

    run = wandb.init(project="test")
    conf: wandb.Config = run.config

    conf.env_id = arg.env_id
    conf.max_step = arg.max_step
    conf.seed = arg.seed
    conf.gamma = arg.gamma

    # fix seed
    if arg.seed is not None:
        manual_seed(arg.seed)

    # make environment
    env = make_env(arg.env_id, arg.seed, arg.save_video_interval is not None)

    sac_agent = SacAgent.configure_agent_from_gym(env, gamma=arg.gamma)

    pprint(sac_agent.config)

    # initialize
    total_step = 0
    total_episode_num = 0

    state = env.reset()
    step = 0
    done = False
    reward_sum = 0

    if arg.save_video_interval is not None:
        env.start_recording()

    print("sampling experience through random actions")
    while total_step < arg.t_init:
        # sample action
        action = env.action_space.sample()

        # enviroment interaction
        next_state, reward, done, _ = env.step(action)
        total_step += 1
        step += 1
        reward_sum += reward

        # check is state terminal
        terminal = is_state_terminal(env, step, done)

        # observe
        sac_agent.observe(state, next_state, action, reward, terminal)

        # update state
        state = next_state

        if done:

            state = env.reset()
            step = 0
            done = False
            reward_sum = 0

            log_data = {"reward_sum": reward_sum}

            if not env.is_frames_empty():
                log_data.update(
                    {
                        "video/random_action": wandb.Video(
                            env.get_frames_array(), fps=60, format="mp4"
                        )
                    }
                )

            wandb.log(log_data, step=total_step)

            total_episode_num += 1
    print("done")

    print("start training")
    while total_step < conf.max_step:
        done = False

        state = env.reset()
        step = 0

        reward_sum = 0

        while not done:

            action = sac_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            terminal = is_state_terminal(env, step, done)

            sac_agent.observe(state, next_state, action, reward, terminal)
            sac_agent.update()

            state = next_state

            reward_sum += reward
            total_step += 1


if __name__ == "__main__":
    train_sac()
