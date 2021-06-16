import copy
import colorama
from colorama import Fore
import torch
from torch import nn
from torch.optim import Adam
import gym
import rlrl.utils.wandb as rlwandb
from rlrl.replay_buffers import ReplayBuffer
from rlrl.q_funcs import ClippedDoubleQF, QFStateAction, delay_update
from rlrl.policies import SquashedGaussianPolicy  # noqa: F401
from rlrl.nn import build_simple_linear_sequential, Lambda
from rlrl.utils import set_global_seed, get_env_info, batch_shaping
import rlrl.agents.sac_agent as sac
import wandb

# import matplotlib.pyplot as plt
# gym.logger.set_level(40)


CHECKPOINT_INTERVAL = 15


def main(config):
    (
        dev,
        env,
        env_info,
        D,
        policy,
        policy_optimizer,
        cdq,
        cdq_t,
        qf_optimizer,
        alpha,
        alpha_optimizer,
        target_entropy,
    ) = make_sac(config)

    total_step = 0

    # log
    wandb.watch(policy)
    wandb.watch(cdq)
    wandb.watch(alpha)

    for epi in range(config.max_episode):
        env_seed = epi
        env.seed(env_seed)
        state = env.reset()
        total_reward = 0
        episode_step = 0
        episode_reward = 0
        episode_action_entropy = 0

        done = False

        while not done:
            wandb.log({"total_step": total_step})
            episode_step += 1
            total_step += 1

            if len(D) < config.t_init:
                action = env.action_space.sample()
            else:
                action, entropy = get_action_and_entropy(state, policy, dev)
                episode_action_entropy += entropy

            state_next, reward, done, _ = env.step(action)
            episode_reward += reward
            terminal = done if episode_step < env_info.max_episode_steps else False
            total_reward += reward
            D.append(sac.Transition(state, action, state_next, reward, not terminal))
            state = state_next  # update state

            if len(D) == config.t_init:
                break

            if total_step > config.t_init:
                Dsub = D.sample(config.batch_size)
                Dsub = batch_shaping(Dsub, torch.cuda.FloatTensor)

                jq = sac.calc_q_loss(Dsub, policy, alpha(), config.gamma, cdq, cdq_t)
                optimize(jq, qf_optimizer)

                jp = sac.calc_policy_loss(Dsub, policy, alpha(), cdq)
                optimize(jp, policy_optimizer)

                jalpha = sac.calc_temperature_loss(Dsub, policy, alpha(), target_entropy)
                optimize(jalpha, alpha_optimizer)
                wandb.log(
                    {
                        "jq": jq,
                        "jp": jp,
                        "jalpha": jalpha,
                        "log_alpha": torch.log(alpha()),
                    },
                    commit=False,
                )
                delay_update(cdq, cdq_t, config.tau)

        wandb.log(
            {
                "episode": epi,
                "EpisodeReward": episode_reward,
                "EpisodeEntropy": episode_action_entropy / episode_step,
            },
            commit=False,
        )

        print(
            f"Epi: {epi}, "
            f"Reward: {total_reward}, "
            f"entropy: {episode_action_entropy / episode_step}, "
            f"alpha: {alpha()}"
        )

        if epi % 30 == 0:
            policy_test(env, policy, dev)


def optimize(j, opti):
    opti.zero_grad()
    j.backward()
    opti.step()


def make_sac(config):

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = gym.make(config.env_id)
    env_info = get_env_info(env)

    set_global_seed(env, config.seed)

    # Replay Memory
    D = ReplayBuffer(config.replay_buffer_capacity)

    # # policy
    # policy_n = build_simple_linear_sequential(env_info.dim_state,
    #                                   env_info.dim_action * 2,
    #                                   **config.policy_n)
    # policy = SquashedGaussianPolicy(policy_n).to(dev)

    # policy
    def squashed_diagonal_gaussian_head(x):
        mean, log_scale = torch.chunk(x, 2, dim=x.dim() // 2)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution
        return torch.distributions.transformed_distribution.TransformedDistribution(
            base_distribution,
            [torch.distributions.transforms.TanhTransform(cache_size=1)],
        )

    policy = torch.nn.Sequential(
        nn.Linear(env_info.dim_state, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, env_info.dim_action * 2),
        Lambda(squashed_diagonal_gaussian_head),
    ).to(dev)

    policy[2].weight.detach().mul_(1e-1)

    policy_optimizer = Adam(policy.parameters(), lr=config.lr)

    # QFunction
    q_net1 = build_simple_linear_sequential(env_info.dim_state + env_info.dim_action, 1, **config.q_n)
    q_net2 = build_simple_linear_sequential(env_info.dim_state + env_info.dim_action, 1, **config.q_n)
    cdq = ClippedDoubleQF(QFStateAction(q_net1), QFStateAction(q_net2)).to(dev)
    cdq_t = copy.deepcopy(cdq)
    qf_optimizer = Adam(cdq.parameters(), lr=config.lr)

    # alpha
    alpha = sac.TemperatureHolder()
    alpha_optimizer = Adam(alpha.parameters(), lr=config.lr)
    target_entropy = -env_info.dim_action

    return (
        dev,
        env,
        env_info,
        D,
        policy,
        policy_optimizer,
        cdq,
        cdq_t,
        qf_optimizer,
        alpha,
        alpha_optimizer,
        target_entropy,
    )


def get_action_and_entropy(state, policy, dev):
    with torch.no_grad():
        # pylint: disable-msg=not-callable,line-too-long
        policy_distrib = policy(torch.tensor(state, dtype=torch.float32, device=dev))
        action = policy_distrib.sample()
        action_log_prob = policy_distrib.log_prob(action)
        action_log_prob = action_log_prob.cpu().numpy()
        action = action.cpu().numpy()

    return action, -action_log_prob


def policy_test(env, policy, dev):
    state = env.reset()
    total_reward = 0
    episode_reward = 0
    episode_action_entropy = 0

    done = False

    vid = rlwandb.GymVideoWandb(env)

    while not done:
        vid.capture_frame()
        action, entropy = get_action_and_entropy(state, policy, dev)
        episode_action_entropy += entropy

        state_next, reward, done, _ = env.step(action)
        episode_reward += reward
        total_reward += reward
        state = state_next  # update state

    wandb.log({"video": vid.get_video()}, commit=False)


if __name__ == "__main__":
    colorama.init(autoreset=True)

    # ENVID = "MountainCarContinuous-v0"
    # ENVID = "BipedalWalkerHardcore-v3"
    # ENVID = "BipedalWalker-v3"
    ENVID = "Swimmer-v2"

    print(f"Gym Enviroment is {Fore.BLUE}{ENVID}")

    conf = dict(
        env_id=ENVID,
        seed=5,
        max_episode=1500,
        q_n={"hidden_units": [256, 256], "hidden_activation": "ReLU"},
        policy_n={"hidden_units": [256, 256], "hidden_activation": "ReLU"},
        tau=5e-3,
        t_init=10000,
        replay_buffer_capacity=1e6,
        batch_size=256,
        gamma=0.995,
        lr=1e-3,
    )

    wandb.init(project=ENVID, name="SoftActorCritic", config=conf)

    conf = wandb.config

    main(conf)
