import torch.nn as nn


def get_module_device(m: nn.Module):
  return next(m.parameters()).device
