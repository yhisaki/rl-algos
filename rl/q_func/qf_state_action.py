import numpy as np
import torch
import torch.nn as nn


class QNStateAction(nn.Module):
  def __init__(self, dim_state: int, dim_action: int, hidden_dim: int):
    super(QNetwork, self).__init__()
    self.dim_state_ = dim_state
    self.dim_action_ = dim_action

    layers = [
        nn.Linear(self.dim_state_ + self.dim_action_, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    ]

    self.Q_ = nn.Sequential(*layers)

  def forward(self, state, action):
    state_action = torch.cat([state, action], dim=1)
    return self.Q_(state_action)


def SynchronizeQNetwork(src: QNetwork, dst: QNetwork, tau: float = 1.0):
  """Synchronize Q Network Parameter.
  tau != 1.0のときはdelay update

  Args:
      src (QNetwork): [description]
      dst (QNetwork): [description]
      tau (float, optional): [description]. Defaults to 1.0.
  """
  for src_param, dst_param in zip(src.parameters(), dst.parameters()):
    dst_param.data.copy_(tau * src_param.data + (1.0-tau)*dst_param.data)
