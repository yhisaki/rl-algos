from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Union

import torch
from torch import distributions, nn


class StochasticHeadBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.is_stochastic = True

    @abstractmethod
    def forward_stochastic(self, *args, **kwargs) -> distributions.Distribution:
        raise NotImplementedError()

    @abstractmethod
    def forward_determistic(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, *args, **kwargs) -> Union[torch.Tensor, distributions.Distribution]:
        if self.is_stochastic:
            return self.forward_stochastic(*args, **kwargs)
        else:
            return self.forward_determistic(*args, **kwargs)

    @contextmanager
    def deterministic(self):
        try:
            pre_state = self.is_stochastic
            self.is_stochastic = False
            yield self
        finally:
            self.is_stochastic = pre_state

    @contextmanager
    def stochastic(self):
        try:
            pre_state = self.is_stochastic
            self.is_stochastic = False
            yield self
        finally:
            self.is_stochastic = pre_state
