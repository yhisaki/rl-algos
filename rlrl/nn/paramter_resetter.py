import torch


@torch.no_grad()
def parameter_resetter(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()
