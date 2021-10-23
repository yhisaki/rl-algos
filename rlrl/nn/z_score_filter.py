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
        return self._var.clone()

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    def update(self, x):
        pass

    def forward(self, x, update=False):
        if update:
            self.update(x)
        normalized = (x - self.mean) * self._std_inverse()
        return normalized


if __name__ == "__main__":
    filter = ZScoreFilter()
