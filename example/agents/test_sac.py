from rlrl.agents import SacAgent, SquashedDiagonalGaussianHead
from rlrl.nn.stochanic_head import determistic, to_determistic, to_stochanic
from rlrl.wrappers import CastObservationToFloat32
import gym
from torch import nn
import torch
from torch import distributions
from torch.distributions import distribution
from torch.nn.modules import activation
import numpy as np
from gym.spaces import Box
from pprint import pprint


if __name__ == "__main__":

    env = gym.make("Swimmer-v2")
    # env = gym.make("Pendulum-v0")

    sac_agent = SacAgent.configure_agent_from_gym(env, device="cuda")

    pprint(sac_agent.config)

    state = env.reset()

    # print(type(env.action_space.high))

    # h = SquashedDiagonalGaussianHead()

    # sac_agent = SacAgent.configure_default_agent(dim_state, dim_action)

    # sac_agent.save("output/test")

    # state = env.reset()
    # with torch.no_grad():
    #     action: distributions.Distribution = sac_agent.policy(torch.tensor(state, device="cuda"))

    # print(action.rsample().requires_grad)

    # sac_agent = SacAgent.configure_default_agent(dim_state=dim_state, dim_action=dim_action)
    # sac_agent.policy.to("cuda")
    # sac_agent.policy.cpu()
    # print(sac_agent.policy.to())

    # loc = torch.zeros(3)
    # scale = torch.ones(3)
    # normal = distributions.Normal(loc, scale)
    # print(normal.sample())
    # policy.apply(to_stochanic)

    # state = torch.tensor(env.observation_space.sample().astype(np.float32)).to("cuda")

    # distrib = policy(state)
    # # print(torch.tensor(env.observation_space.sample()))
    # # print(sdgh.scale * torch.tensor(env.observation_space.sample()))
    # action = distrib.sample()
    # print(action)
    # print(distrib.log_prob(action))
    # sdgh = SquashedDiagonalGaussianHead()
