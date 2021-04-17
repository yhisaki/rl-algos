import torch
import torch.nn as nn


class QFStateAction(nn.Module):
  """状態行動対から行動価値を推定する．

  今のところ，ネットワークの構造は2層で活性化関数はReLuで固定．
  コンストラクタで隠れ層の次元を指定できる．

  Args:
      nn ([type]): [description]
  """

  def __init__(self, dim_state: int, dim_action: int, hidden_dim: int):
    super(QFStateAction, self).__init__()
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


def delay_update(src: QFStateAction, dst: QFStateAction, tau: float = 1.0):
  """delay update qf.

  Args:
      src (QNetwork): [description]
      dst (QNetwork): [description]
      tau (float, optional): [description]. Defaults to 1.0.
  """
  for src_param, dst_param in zip(src.parameters(), dst.parameters()):
    dst_param.data.copy_(tau * src_param.data + (1.0 - tau) * dst_param.data)
