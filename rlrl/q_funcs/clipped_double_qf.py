import copy
import torch
import torch.nn as nn


class ClippedDoubleQF(nn.Module):
  def __init__(self, qf):
    super(ClippedDoubleQF, self).__init__()
    self.qf1 = copy.deepcopy(qf)
    self.qf2 = copy.deepcopy(qf)

  def forward(self, *args):
    return torch.min(self.qf1.forward(*args), self.qf2.forward(*args))
