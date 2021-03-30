import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


EPSILON = 1e-6


class GaussianPolicy(nn.Module):
  def __init__(self, device, dim_state: int, dim_action: int, hidden_dim: int,
               action_high: np.ndarray, action_low: np.ndarray, log_std_bounds=(2, -20)):
    super(GaussianPolicy, self).__init__()
    self.dim_state_ = dim_state
    self.dim_action_ = dim_action

    # 方策のエントロピーが大きくなりすぎたり小さくなりすぎるのを防ぐために範囲を指定
    self.log_std_max, self.log_std_min = log_std_bounds

    # actionをaction_high > a > action_lowにおさめるため
    self.action_bias_ = torch.FloatTensor((action_high + action_low) / 2.0).to(device)
    self.action_scale_ = torch.FloatTensor((action_high - action_low) / 2.0).to(device)

    shared_layers = [
        nn.Linear(dim_state, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU()
    ]

    self.shared_layers_ = nn.Sequential(*shared_layers)
    self.mean_layer_ = nn.Linear(hidden_dim, dim_action)
    self.log_std_layer_ = nn.Linear(hidden_dim, dim_action)

    self.to(device)
    self.dev_ = device

  def forward(self, state):
    out = self.shared_layers_(state)
    loc = self.mean_layer_(out)
    log_std = self.log_std_layer_(out)
    log_std = log_std.clamp(min=self.log_std_min, max=self.log_std_max)
    return loc, log_std

  def sample(self, state):
    """a~πΦ(・|s)となる確率変数aをサンプルしその対数尤度logπΦ(a|s)をもとめる．戻り値は微分可能

    Args:
        state ([type]): 状態量

    Returns:
        action, log_prob [type]: [description]
    """
    change_numpy_and_1d = False
    if isinstance(state, np.ndarray) and (state.ndim == 1):
      change_numpy_and_1d = True
      state = torch.Tensor([state])
    
    state = state.to(self.dev_)
    mean, log_std = self.forward(state)
    std = log_std.exp()  # 0 < std
    normal = Normal(mean, std)
    _u = normal.rsample()  # noise, for reparameterization trick (mean + std * N(0,1))
    _action = torch.tanh(_u)  # -1 < _action < 1, スケーリング前
    # action_low < action < action_high
    action = _action * self.action_scale_ + self.action_bias_

    # actionの対数尤度をもとめる
    # tanhとかscale, biasで補正した分normal.log_prob(u)だけでは求まらない
    log_prob = normal.log_prob(_u).sum(dim=1)
    log_prob -= 2 * \
        (self.action_scale_ * (np.log(2) - _u -
                               F.softplus(-2 * _u))).sum(dim=1)

    if change_numpy_and_1d:
      action = action[0].detach()
      log_prob = log_prob[0].detach()
    return action, log_prob
