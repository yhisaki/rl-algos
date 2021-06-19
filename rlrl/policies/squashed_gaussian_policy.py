import copy
from contextlib import contextmanager
from rlrl.nn.lmbda import Lambda
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
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )


def squashed_diagonal_gaussian_head(x):
    mean, log_scale = torch.chunk(x, 2, dim=x.dim() // 2)
    log_scale = torch.clamp(log_scale, -20.0, 2.0)
    var = torch.exp(log_scale * 2)
    base_distribution = distributions.Independent(
        distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
    )
    # cache_size=1 is required for numerical stability
    # https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution
    return distributions.transformed_distribution.TransformedDistribution(
        base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
    )


def determistic_squashed_gaussian_policy_head(x):
    mean, _ = torch.chunk(x, 2, dim=x.dim() // 2)
    return torch.tanh(mean)


@contextmanager
def determistic_squashed_gaussian_policy(policy: nn.Sequential):
    """Temporaliy switch squashed gaussian policy to determistic policy

    Args:
        policy (nn.Sequential): [description]

    Yields:
        [type]: [description]
    """
    try:
        del policy[-1]
        head = Lambda(determistic_squashed_gaussian_policy_head)
        policy.add_module(
            head.__repr__(),
            head,
        )
        yield policy
    finally:
        del policy[-1]
        head = Lambda(squashed_diagonal_gaussian_head)
        policy.add_module(
            head.__repr__(),
            head,
        )
