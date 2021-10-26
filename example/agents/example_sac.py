import argparse
import logging

import wandb
from rlrl.agents import SacAgent
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.experiments import GymMDP
from rlrl.wrappers import make_envs_for_training


def train_sac():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="Swimmer-v2", type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--max_step", default=int(1e6), type=int)
    parser.add_argument("--gamma", default=0.995, type=float)
    parser.add_argument("--t_init", default=int(1e4), type=int)
    parser.add_argument("--save_video_interval", default=None, type=int)
    parser.add_argument("--save_agent", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run = wandb.init(project="rlrl_example", name="soft_actor_critic", tags=[args.env_id])
    conf: wandb.Config = run.config

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

    # make agent
    sac_agent = SacAgent(
        dim_state=dim_state,
        dim_action=dim_action,
        gamma=args.gamma,
        num_random_act=args.t_init,
    )

    # save conf
    conf.env_id = args.env_id
    conf.seed = args.seed
    conf.max_step = args.max_step
    conf.gamma = sac_agent.gamma
    conf.agent_device = sac_agent.device

    def actor(state):
        return sac_agent.act(state)

    interactions = GymMDP(env, actor, max_step=args.max_step)
    for step, states, next_states, actions, rewards, dones in interactions:
        sac_agent.observe(
            states,
            next_states,
            actions,
            rewards,
            is_state_terminal(env, step, dones),
        )


if __name__ == "__main__":
    # nohup python3 -u rlrl/example/agents/example_sac.py --env_id "Swimmer-v2" --save_video_interval 5000 --seed 0 --gamma 0.997 --save_agent True &  # noqa: E501
    train_sac()
