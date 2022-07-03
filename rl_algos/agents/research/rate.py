import torch
from torch import nn
from torch.optim import Adam


class RateHolder(nn.Module):
    def __init__(self, initial_rate: float = 0.0):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(initial_rate, dtype=torch.float32))

    def forward(self):
        return self.rate


def default_rate_initializer(device: torch.device):
    rate = RateHolder().to(device)
    opti = Adam(rate.parameters(), lr=1e-5)
    return rate, opti
