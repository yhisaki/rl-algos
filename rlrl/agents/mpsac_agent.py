import copy
import numpy as np
import torch
import gym
from rlrl.q_funcs.qf_state_action import QFStateAction
from rlrl.q_funcs.clipped_double_qf import ClippedDoubleQF
from rlrl.replay_buffers.replay_buffer import ReplayBuffer
from rlrl.policies.gaussian_policy import GaussianPolicy


def construct_mpsac_from_json(j):
  env = gym.make(j["env_name"])
  dim_state = env.observation_space.shape[0]
  dim_action = env.action_space.shape[0]
  pass


class MpSacAgent():
  def __init__(self,
               dim_state: int, dim_action: int,
               high_action: np.ndarray, low_action: np.ndarray,
               hidden_dim: int,
               t_init: int,
               D):
    self.dim_state_ = dim_state
    self.dim_action_ = dim_action

    self.high_action_ = high_action
    self.low_action_ = low_action

    self.total_step_ = 0
    self.total_epi_ = 0

    # 最初Tinit回はランダムに行動する．
    self.t_init_ = t_init

    # 方策の個数
    self.num_policies = 2

    # 勾配による更新に用いるq関数
    self.qfunctions_ = [ClippedDoubleQF(QFStateAction(self.dim_state_, self.dim_action_, hidden_dim)) for _ in range(self.num_policies)]
    # Delay Update のためのターゲットネットワーク.
    # 勾配によるパラメーター更新することは無い
    self.qftargets_ = [ copy.deepcopy(q).eval().requires_grad_(False) for q in self.qfunctions_]

    self.policy_ = GaussianPolicy(
        dim_state, dim_action, hidden_dim, high_action, low_action)

    self.policies_ = []

  def update(self, state):
    if self.total_step_ < self.t_init_:
      action = self.sample_random_action()
    else:
      policy_idx = self.total_epi_ % self.num_policies
      action, action_log_prob = self.policies_[policy_idx].sample(state)
      actions = []
      action_log_probs = []
      # for policy in self.policy_:

      pass

    self.total_step_ += 1
    return action

  def sample_random_action(self):
    return np.random.uniform(self.low_action_, self.high_action_)
  # def _cal_current_
