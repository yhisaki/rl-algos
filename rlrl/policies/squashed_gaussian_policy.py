import copy
import torch
import torch.nn as nn
from torch import distributions


class SquashedGaussianPolicy(nn.Module):
    def __init__(self, layer):

        super(SquashedGaussianPolicy, self).__init__()

        self._layer = copy.deepcopy(layer)

    def forward(self, state):
        x = self._layer(state)
        mean, log_scale = torch.chunk(x, 2, dim=x.dim() // 2)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1)
        # https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )


def squashed_diagonal_gaussian_head(x):
    mean, log_scale = torch.chunk(x, 2, dim=x.dim() // 2)
    log_scale = torch.clamp(log_scale, -20.0, 2.0)
    var = torch.exp(log_scale * 2)
    base_distribution = distributions.Independent(distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1)
    # cache_size=1 is required for numerical stability
    return distributions.transformed_distribution.TransformedDistribution(base_distribution, [distributions.transforms.TanhTransform(cache_size=1)])
