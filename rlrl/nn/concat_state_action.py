from typing import Tuple

import torch
from torch import nn


class ConcatStateAction(nn.Module):
    def __init__(self):
        super(ConcatStateAction, self).__init__()

    def forward(self, state_action: Tuple[torch.Tensor, torch.Tensor]):
        return torch.cat(state_action, dim=-1)
