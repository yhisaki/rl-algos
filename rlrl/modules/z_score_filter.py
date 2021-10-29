import torch
from torch import nn


class ZScoreFilter(nn.Module):
    def __init__(self, size, eps=1e-2):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(size, dtype=torch.float32))
        self.register_buffer("_var", torch.ones(size, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0))
        self.eps = eps
        self._cached_std_inverse = None

    @property
    def mean(self):
        return self._mean.clone()

    @property
    def std(self):
        return torch.sqrt(self._var).clone()

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    def update(self, x: torch.Tensor):
        count_x = x.shape[0]
        if count_x == 0:
            return

        self.count += count_x
        rate = count_x / self.count.float()
        assert rate > 0
        assert rate <= 1

        var_x, mean_x = torch.var_mean(x, axis=0, keepdim=True, unbiased=False)
        var_x = var_x.squeeze()
        mean_x = mean_x.squeeze()
        delta_mean = mean_x - self._mean
        self._mean += (rate * delta_mean).squeeze()
        self._var += (rate * (var_x - self._var + delta_mean * (mean_x - self._mean))).squeeze()

        # clear cache
        self._cached_std_inverse = None

    def forward(self, x, update=False):
        if update:
            self.update(x)
        normalized = (x - self.mean) * self._std_inverse
        return normalized
