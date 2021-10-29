from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Union

import torch
from torch import distributions, nn


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

    @contextmanager
    def deterministic(self):
        try:
            pre_state = self.is_stochanic
            self.is_stochanic = False
            yield self
        finally:
            self.is_stochanic = pre_state

    @contextmanager
    def stochanic(self):
        try:
            pre_state = self.is_stochanic
            self.is_stochanic = False
            yield self
        finally:
            self.is_stochanic = pre_state
