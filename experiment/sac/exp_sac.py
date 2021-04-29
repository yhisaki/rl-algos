import json
import copy
import argparse
import torch
from torch import nn # pylint: disable=unused-import
from torch.optim import Adam
import gym
from rlrl.replay_buffers import ReplayBuffer
from rlrl.q_funcs import ClippedDoubleQF, QFStateAction, delay_update
from rlrl.policies import SquashedGaussianPolicy
from rlrl.nn import build_simple_linear_nn
from rlrl.utils import (
    set_global_torch_device,
    get_global_torch_device,
    set_global_seed,
    get_env_info,
    batch_shaping
)
import rlrl.agents.sac_agent as sac
# gym.logger.set_level(40)


def main(j):
  parser = argparse.ArgumentParser()

  # make enviroment
  env = gym.make(j["env_id"])
  env_info = get_env_info(env)

  set_global_seed(env, j["seed"])

  # set deviceh
  set_global_torch_device(j["device"])

  # Replay Memory
  D = ReplayBuffer(j["replay_buffer_capacity"])

  # policy
  policy_n = build_simple_linear_nn(env_info.dim_state, env_info.dim_action * 2,
                                    j["policy_n"]["hidden_unit"], eval(j["policy_n"]["hidden_activation"]))
  policy = SquashedGaussianPolicy(policy_n).to(get_global_torch_device())
  policyOptimizer = Adam(policy.parameters(), lr=j["lr"])

  # QFunction
  cdq = ClippedDoubleQF(QFStateAction, *make_two_qf(env_info, j)).to(get_global_torch_device())
  cdq_t = copy.deepcopy(cdq)
  qfOptimizer = Adam(cdq.parameters(), lr=j["lr"])

  # alpha
  alpha = sac.TemperatureHolder()
  alphaOptimizer = Adam(alpha.parameters(), lr=j["lr"])
  target_entropy = - env_info.dim_action

  # log
  total_step = 0

  for epi in range(j["episodes_num"]):
    env.seed(epi)
    state = env.reset()
    episode_reward = 0
    episode_step = 0
    episode_action_entropy = 0

    while True:
      episode_step += 1
      total_step += 1

      if len(D) < j["t_init"]:
        action = env.action_space.sample()
      else:
        with torch.no_grad():
          action, entropy = get_action_and_entropy(state, policy)
          episode_action_entropy += entropy

      state_next, reward, done, _ = env.step(action)
      terminal = done if episode_step < env_info.max_episode_steps else False
      episode_reward += reward
      D.append(sac.Transition(state, action, state_next, reward, not terminal))
      # update state
      state = state_next
      if total_step > j["t_init"]:
        # learning
        Dsub = D.sample(j["batch_size"])
        Dsub = batch_shaping(Dsub, torch.cuda.FloatTensor)

        jq = sac.calc_q_loss(Dsub, policy, alpha(), j["gamma"], cdq, cdq_t)
        optimize(jq, qfOptimizer)

        jp = sac.calc_policy_loss(Dsub, policy, alpha(), cdq)
        optimize(jp, policyOptimizer)

        jalpha = sac.calc_temperature_loss(Dsub, policy, alpha(), target_entropy)
        optimize(jalpha, alphaOptimizer)

        delay_update(cdq, cdq_t, j["tau"])

      if done:
        print(f"Epi: {epi}, Reward: {episode_reward}, entropy: {episode_action_entropy / episode_step}, alpha: {alpha()}")
        break


def optimize(j, opti):
  opti.zero_grad()
  j.backward()
  opti.step()


def make_two_qf(env_info, j):
  q_net1 = build_simple_linear_nn(env_info.dim_state + env_info.dim_action, 1,
                                  j["q_n"]["hidden_unit"], eval(j["q_n"]["hidden_activation"]))
  q_net2 = build_simple_linear_nn(env_info.dim_state + env_info.dim_action, 1,
                                  j["q_n"]["hidden_unit"], eval(j["q_n"]["hidden_activation"]))
  return q_net1, q_net2


def get_action_and_entropy(state, policy):
  with torch.no_grad():
    # pylint: disable-msg=not-callable,line-too-long
    policy_disturb = policy(torch.tensor(state, dtype=torch.float32, device=get_global_torch_device()))
    action = policy_disturb.sample()
    action_log_prob = policy_disturb.log_prob(action)
    action_log_prob = action_log_prob.cpu().numpy()
    action = action.cpu().numpy()

  return action, - action_log_prob


if __name__ == '__main__':
  # load json file

  path_to_parameter = "experiment/sac/parameters/MountainCarContinuous-v0.sac.param.json"
  # path_to_parameter = "experiment/sac/parameters/BipedalWalker-v3.sac.param.json"

  f = open(path_to_parameter, mode='r')
  param = json.load(f)

  print(f'Gym Enviroment is \033[34m{param["env_id"]}\033[0m')

  main(param)
