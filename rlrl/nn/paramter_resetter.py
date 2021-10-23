import torch
from torch import nn


@torch.no_grad()
def parameter_resetter(m):
    if not isinstance(m, nn.Module):
        raise ValueError("m must be torch.nn.Module")
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()
