import copy
import torch
import torch.nn as nn


class ClippedDoubleQF(nn.Module):
  def __init__(self, QF, *args, **kwargs):
    super(ClippedDoubleQF, self).__init__()
    self.qf1 = QF(*args, **kwargs)
    self.qf2 = QF(*args, **kwargs)

  def forward(self, *args):
    return torch.min(self.qf1.forward(*args), self.qf2.forward(*args))

  def __getitem__(self, idx: int):
    if idx == 0:
      return self.qf1
    elif idx == 1:
      return self.qf2
    else:
      IndexError("ClippedDoubleQF Index Out Of Range")