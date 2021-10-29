from rlrl.modules.distributions.stochanic_head_base import StochanicHeadBase
import torch
from torch import nn


class GaussianHeadWithStateIndependentCovariance(StochanicHeadBase):
    def __init__(self, dim_action):
        super().__init__()
        self.action_log_std = nn.Parameter(torch.zeros(dim_action))

    def forward_stochanic(self, x):
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=x, scale=torch.exp(self.action_log_std)), 1
        )

    def forward_determistic(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SquashedDiagonalGaussianHead(StochanicHeadBase):
    def __init__(self):
        super().__init__()

    def forward_stochanic(self, x):
        mean, log_scale = torch.chunk(x, 2, dim=x.dim() // 2)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        # https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution
        return torch.distributions.transformed_distribution.TransformedDistribution(
            base_distribution,
            [
                torch.distributions.transforms.TanhTransform(cache_size=1),
            ],
        )

    def forward_determistic(self, x):
        mean, _ = torch.chunk(x, 2, dim=x.dim() // 2)
        return torch.tanh(mean)
