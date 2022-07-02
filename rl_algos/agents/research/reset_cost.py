import torch
from torch import nn
import torch.nn.functional as F


class ResetCostHolder(nn.Module):
    def __init__(self, initial_reset_cost_param: float = 0.0):
        super().__init__()
        self.reset_cost_param = nn.Parameter(
            torch.tensor(initial_reset_cost_param, dtype=torch.float32)
        )

    def forward(self):
        return F.softplus(self.reset_cost_param)
