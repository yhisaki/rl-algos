import torch.nn as nn


def get_module_device(m: nn.Module):
    """Find out which device the torch module is on.

    Args:
        m (nn.Module)

    Returns:
        device
    """
    return next(m.parameters()).device
