from typing import List

import torch
import torch.nn as nn


def _all_the_same(elements):
    return elements[1:] == elements[:-1]


def get_module_device(m: nn.Module):
    """Find out which device the torch module is on.

    Args:
        m (nn.Module)

    Returns:
        device
    """
    devices: List[torch.device] = []
    for p in m.parameters():
        devices.append(p.device)

    if not _all_the_same(devices):
        raise ValueError("Not all parameters are placed on the same device.")

    return devices[0]
