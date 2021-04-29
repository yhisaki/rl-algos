import torch
import torch.nn as nn


class ClippedDoubleQF(nn.Module):
  def __init__(self, QF, qn1, qn2):
    """[summary]

    Args:
        QF : class of Q Function
        qn1 : argument of first Q Function
        qn2 : argument of first Q Function
    """
    super(ClippedDoubleQF, self).__init__()
    self.qf1 = QF(qn1)
    self.qf2 = QF(qn2)

  def forward(self, *args):
    return torch.min(self.qf1.forward(*args), self.qf2.forward(*args))

  def __getitem__(self, idx: int):
    if idx == 0:
      return self.qf1
    elif idx == 1:
      return self.qf2
    else:
      IndexError("ClippedDoubleQF Index Out Of Range")
