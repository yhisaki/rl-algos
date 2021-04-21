import json
import argparse
import numpy as np
import torch
from torch.optim import Adam
import gym
from gym import wrappers
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
from rlrl.agents.sac_agent import *

# gym.logger.set_level(40)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # load json file
  f = open('experiment/sac/MountainCarContinuous-v0.sac.param.json', mode='r')
  j = json.load(f)
  print(f'Gym Enviroment is \033[34m{j["env_name"]}\033[0m')

  # make enviroment
  env = gym.make(j["env_name"])
  env_info = get_env_info(env)
  # env = wrappers.Monitor(env, "log/video/" + j["env_name"], video_callable=(lambda ep: ep % 50 == 0), force=True)

  set_global_seed(env, 4)

  # set deviceh
  set_global_torch_device('cuda')

  # Replay Memory
  D = ReplayBuffer(j["replay_buffer_capacity"])

  # policy
  policy = GaussianPolicy(env_info, j["hidden_dim"]).to(get_global_torch_device())
  policyOptimizer = Adam(policy.parameters(), lr=j["lr"])

  # QFunction
  cdq = ClippedDoubleQF(QFStateAction, env_info, j["hidden_dim"]).to(get_global_torch_device())
  cdq_t = copy.deepcopy(cdq)
  qfOptimizer = Adam(cdq.parameters(), lr=j["lr"])

  # alpha
  alpha = TemperatureHolder()
  alphaOptimizer = Adam(alpha.parameters(), lr=j["lr"])
  target_entropy = - env_info.dim_action

  # log
  total_step = 0

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
          action, action_log_prob = policy.sample(torch.cuda.FloatTensor([state]))
          action = action.cpu().numpy().squeeze(1)
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

        jq = calc_q_loss(Dsub, policy, alpha.to_float(), j["gamma"], cdq, cdq_t)
        qfOptimizer.zero_grad()
        jq.backward()
        qfOptimizer.step()

        jp = calc_policy_loss(Dsub, policy, alpha.to_float(), cdq)
        policyOptimizer.zero_grad()
        jp.backward()
        policyOptimizer.step()

        jalpha = calc_temperature_loss(Dsub, policy, alpha(), target_entropy)
        alphaOptimizer.zero_grad()
        jalpha.backward()
        alphaOptimizer.step()

        delay_update(cdq, cdq_t, j["tau"])

      if done:
        print(f"Epi: {epi}, Reward: {episode_reward}, action_log_prob: {action_log_prob_total / episode_step}, alpha: {alpha()}")
        break
