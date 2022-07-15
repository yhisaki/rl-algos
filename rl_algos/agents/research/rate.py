import torch
from torch import nn


class RateHolder(nn.Module):
    def __init__(self, initial_rate: float = 0.0):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(initial_rate, dtype=torch.float32))

    def forward(self):
        return self.rate


def default_rate_fn():
    rate = RateHolder()
    return rate
