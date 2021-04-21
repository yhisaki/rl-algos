import json
import argparse
import numpy as np
import torch
from torch.optim import Adam
import gym
from rlrl.replay_buffers import ReplayBuffer
from rlrl.q_funcs import ClippedDoubleQF, QFStateAction, delay_update
from rlrl.policies import GaussianPolicy
from rlrl.utils import (
    set_global_torch_device,
    get_global_torch_device,
    set_global_seed,
    get_env_info,
    batch_shaping
)
from rlrl.agents.mp_sac_agent import *
import matplotlib.pyplot as plt

# gym.logger.set_level(40)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # load json file
  f = open('experiment/mp_sac/BipedalWalker-v3.sac.param.json', mode='r')
  j = json.load(f)
  print(f'Gym Enviroment is \033[34m{j["env_name"]}\033[0m')

  # make enviroment
  env = gym.make(j["env_name"])
  env_info = get_env_info(env)
  state = env.reset()

  set_global_seed(env, 5)

  # set deviceh
  set_global_torch_device('cuda')

  # Replay Memory
  D = ReplayBuffer(j["replay_buffer_capacity"])

  # policy
  policy = GaussianPolicy(env_info, j["hidden_dim"]).to(get_global_torch_device())
  policyOptimizer = Adam(policy.parameters(), lr=j["lr"])

  policy2 = copy.deepcopy(policy)
  policyOptimizer2 = Adam(policy2.parameters(), lr=j["lr"])

  # QFunction
  cdq = ClippedDoubleQF(QFStateAction, env_info, j["hidden_dim"]).to(get_global_torch_device())
  cdq_t = copy.deepcopy(cdq)
  qfOptimizer = Adam(cdq.parameters(), lr=j["lr"])

  cdq2 = copy.deepcopy(cdq)
  cdq_t2 = copy.deepcopy(cdq)
  qfOptimizer2 = Adam(cdq2.parameters(), lr=j["lr"])


  # alpha
  log_alpha = torch.zeros(1, requires_grad=True)
  alpha = log_alpha.exp().to(get_global_torch_device())
  alphaOptimizer = Adam([log_alpha], lr=j["lr"])
  target_entropy = - env_info.dim_action
  
  # beta
  log_beta = torch.zeros(1, requires_grad=True)
  beta = log_beta.exp().to(get_global_torch_device())
  betaOptimizer = Adam([log_beta], lr=j["lr"])

  # log
  total_step = 0
  reward_log = []
  reward_log2 = []

  for epi in range(j["episodes_num"]):
    env.seed(epi)
    state = env.reset()
    episode_reward = 0
    episode_step = 0
    action_log_prob_total = 0

    while True:

      episode_step += 1
      total_step += 1

      if len(D) < j["t_init"]:
        action = env.action_space.sample()
      else:
        with torch.no_grad():
          if epi%2==0:
            action, action_log_prob = policy.sample(torch.cuda.FloatTensor([state]))
          else:
            action, action_log_prob = policy2.sample(torch.cuda.FloatTensor([state]))
          action = action.cpu().numpy()[0]
          action_log_prob_total += action_log_prob.cpu().numpy()

      state_next, reward, done, _ = env.step(action)
      terminal = done if episode_step < env_info.max_episode_steps else False
      episode_reward += reward
      D.append(Transition(state, action, state_next, reward, not terminal))
      state = state_next
      if total_step > j["t_init"]:
        # 学習
        Dsub = D.sample(j["batch_size"])
        Dsub = batch_shaping(Dsub, torch.cuda.FloatTensor)

        # Optimize First Policy
        jq = calc_q_loss(Dsub, policy, alpha, j["gamma"], cdq, cdq_t)
        qfOptimizer.zero_grad()
        jq.backward(retain_graph=True)
        qfOptimizer.step()

        jp = calc_policy_loss(Dsub, policy, alpha, cdq)
        policyOptimizer.zero_grad()
        jp.backward(retain_graph=True)
        policyOptimizer.step()

        # Optimize Second Policy
        jq2 = calc_q_loss2(Dsub, policy2, policy, beta, 10*beta, j["gamma"], cdq2, cdq_t2)
        qfOptimizer2.zero_grad()
        jq2.backward(retain_graph=True)
        qfOptimizer2.step()

        jp2 = calc_policy_loss2(Dsub, policy2, policy, beta, 10*beta, cdq2)
        policyOptimizer2.zero_grad()
        jp2.backward(retain_graph=True)
        policyOptimizer2.step()

        jalpha = calc_temperature_loss(Dsub, policy, log_alpha, target_entropy)
        alphaOptimizer.zero_grad()
        jalpha.backward()
        alphaOptimizer.step()
        alpha = log_alpha.exp().to(get_global_torch_device())

        jbeta = calc_temperature_loss(Dsub, policy2, log_beta, target_entropy)
        betaOptimizer.zero_grad()
        jbeta.backward()
        betaOptimizer.step()
        beta = log_alpha.exp().to(get_global_torch_device())

        delay_update(cdq, cdq_t, j["tau"])
        delay_update(cdq2, cdq_t2, j["tau"])

      if done:
        print(f"Epi: {epi}, Reward: {episode_reward}, action_log_prob: {action_log_prob_total / episode_step}, Policy: {'firstpolicy' if epi%2==0 else 'second_policy'}")
        (reward_log if epi%2==0 else reward_log2).append(episode_reward)
        break
  
  plt.plot(reward_log)
  plt.plot(reward_log2)
  plt.show()
