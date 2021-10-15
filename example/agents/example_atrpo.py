import argparse
import logging
# import wandb
from rlrl.utils import manual_seed
from rlrl.experiments import GymMDP
from rlrl.agents.atrpo_agent import AtrpoAgent
from rlrl.nn.z_score_filter import ZScoreFilter  # NOQA
from rlrl.utils import is_state_terminal
from rlrl.wrappers import make_envs_for_training


def train_trpo():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    # parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--env_id", type=str, default="Swimmer-v2")
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    args = parser.parse_args()

    manual_seed(args.seed)

    env = make_envs_for_training(args.env_id, args.num_envs, args.seed)
    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space.shape[-1] if args.num_envs == 1 else env.action_space[0].shape[-1]

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
        lambd=0.97,
        conjugate_gradient_damping=1e-1,
        update_interval=args.num_envs * 1000,
    )

    def actor(state):
        return agent.act(state)

    interactions = GymMDP(env, actor, max_step=1e7)

    for step, states, next_states, actions, rewards, dones in interactions:
        agent.observe(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            terminals=is_state_terminal(env, step, dones),
            resets=dones,
        )


if __name__ == "__main__":
    train_trpo()
