import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


class ResetCostHolder(nn.Module):
    def __init__(self, initial_reset_cost_param: float = 0.0):
        super().__init__()
        self.reset_cost_param = nn.Parameter(
            torch.tensor(initial_reset_cost_param, dtype=torch.float32)
        )

    def forward(self):
        return F.softplus(self.reset_cost_param)


class FixedResetCost(nn.Module):
    def __init__(self, reset_cost) -> None:
        super().__init__()
        self.reset_cost = nn.Parameter(
            torch.tensor(reset_cost, dtype=torch.float32), requires_grad=False
        )
        self.dummy_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self):
        return self.reset_cost + 0.0 * self.dummy_param


def default_reset_cost_fn(device: torch.device):
    rc = ResetCostHolder().to(device)
    opti = Adam(rc.parameters(), lr=1e-3)
    return rc, opti
