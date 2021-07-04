import argparse
from typing import Optional

import gym

import wandb
from rlrl.agents import SacAgent
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.experiments import GymInteractions
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
    parser.add_argument("--save_agent", default=False, type=bool)
    args = parser.parse_args()

    run = wandb.init(project="test_example_sac")
    conf: wandb.Config = run.config

    conf.update(args)

    # fix seed
    if args.seed is not None:
        manual_seed(args.seed)

    # make environment
    env = make_env(args.env_id, args.seed, args.save_video_interval is not None)

    # make agent
    sac_agent = SacAgent.configure_agent_from_gym(env, gamma=args.gamma)

    print(sac_agent)

    # ------------------------ START INTERACTING WITH THE ENVIRONMENT ------------------------
    try:
        total_step = 0
        total_epi = 0

        # Start Sampling
        def random_actor(state):
            return env.action_space.sample()

        if args.save_video_interval is not None:
            env.start_recording()  # save the video wit random action only the first time

        print("sampling experience through random actions")

        interactions = GymInteractions(env, random_actor, max_step=args.t_init)
        for step, state, next_state, action, reward, done in interactions:
            total_step += 1
            terminal = is_state_terminal(env, step, done)
            sac_agent.observe(state, next_state, action, reward, terminal)

            if done:
                total_epi += 1
                log_data = {"reward_sum": interactions.reward_sum}
                if isinstance(env, NumpyArrayMonitor) and not env.is_frames_empty():
                    log_data.update({"video": wandb.Video(env.frames, fps=60, format="mp4")})
                wandb.log(log_data, step=total_step)
        print("sampling is finished\n\n")

        # Start Training
        def agent_actor(state):
            return sac_agent.act(state)

        interactions = GymInteractions(env, agent_actor, max_step=args.max_step)

        for step, state, next_state, action, reward, done in interactions:
            total_step += 1
            terminal = is_state_terminal(env, step, done)
            sac_agent.observe(state, next_state, action, reward, terminal)
            sac_agent.update()

            if done:
                total_epi += 1
                log_data = {
                    "reward_sum": interactions.reward_sum,
                    "loss/q": sac_agent.q1_loss + sac_agent.q2_loss,
                    "loss/policy": sac_agent.policy_loss,
                }
                # if isinstance(env, NumpyArrayMonitor) and not env.is_frames_empty():
                #     log_data.update({"video": wandb.Video(env.frames, fps=60, format="mp4")})
                wandb.log(log_data, step=total_step)
                print(f"Epi : {total_epi}, Reward Sum : {interactions.reward_sum}", flush=True)

    finally:
        if args.save_agent:
            sac_agent.save(wandb.run.dir + "/agent")


if __name__ == "__main__":
    # python3 rlrl/example/agents/example_sac.py --env_id "Pendulum-v0" --seed 0
    # nohup python3 rlrl/example/agents/example_sac.py --env_id "Swimmer-v2" --seed 0 --gamma 0.997 &
    train_sac()
