import argparse

import wandb
from rlrl.agents import SacAgent
from rlrl.utils import is_state_terminal, manual_seed
from rlrl.experiments import GymMDP
from rlrl.wrappers import make_env


def train_sac():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="Swimmer-v2", type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--max_step", default=int(1e6), type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--t_init", default=int(1e4), type=int)
    parser.add_argument("--save_video_interval", default=None, type=int)
    parser.add_argument("--save_agent", action="store_true")
    args = parser.parse_args()

    run = wandb.init(project="rlrl_example", name="soft_actor_critic", tags=[args.env_id])
    conf: wandb.Config = run.config

    # fix seed
    if args.seed is not None:
        manual_seed(args.seed)

    # make environment
    env = make_env(
        args.env_id,
        args.seed,
        monitor=args.save_video_interval is not None,
        monitor_args={"interval_step": args.save_video_interval},
    )

    # make agent
    sac_agent = SacAgent.configure_agent_from_gym(env, gamma=args.gamma)

    # save conf
    conf.env_id = args.env_id
    conf.seed = args.seed
    conf.max_step = args.max_step
    conf.gamma = sac_agent.gamma
    conf.agent_device = sac_agent.device

    print(sac_agent)

    # ------------------------ START INTERACTING WITH THE ENVIRONMENT ------------------------
    try:

        def actor(state):
            return sac_agent.act(state)

        interactions = GymMDP(env, actor, max_step=args.max_step)
        for step, state, next_state, action, reward, done in interactions:
            terminal = is_state_terminal(env, step, done)
            sac_agent.observe(state, next_state, action, reward, terminal)

    finally:
        if args.save_agent:
            sac_agent.save(wandb.run.dir + "/agent")


if __name__ == "__main__":
    # nohup python3 -u rlrl/example/agents/example_sac.py --env_id "Swimmer-v2" --save_video_interval 5000 --seed 0 --gamma 0.997 --save_agent True &  # noqa: E501
    train_sac()
