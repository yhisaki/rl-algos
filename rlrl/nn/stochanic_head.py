from typing import Union
from torch import distributions, nn
from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch


class StochanicHeadBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.is_stochanic = True

    @abstractmethod
    def forward_stochanic(self, *args, **kwargs) -> distributions.Distribution:
        raise NotImplementedError()

    @abstractmethod
    def forward_determistic(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, *args, **kwargs) -> Union[torch.Tensor, distributions.Distribution]:
        if self.is_stochanic:
            return self.forward_stochanic(*args, **kwargs)
        else:
            return self.forward_determistic(*args, **kwargs)


def to_stochanic(m: nn.Module):
    if isinstance(m, StochanicHeadBase):
        m.is_stochanic = True


def to_determistic(m: nn.Module):
    if isinstance(m, StochanicHeadBase):
        m.is_stochanic = False


@contextmanager
def determistic(stochanic_policies):
    istrains = [m.training for m in stochanic_policies]
    try:
        for m in stochanic_policies:
            m.eval()
            yield m
    finally:
        for m, istrain in zip(stochanic_policies, istrains):
            if istrain:
                m.train()
