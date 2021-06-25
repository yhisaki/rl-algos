from torch import nn
from abc import ABC, abstractmethod
from contextlib import contextmanager


class StochanicPolicyHeadBase(nn.Module, ABC):
    def __init__(self):
        super(StochanicPolicyHeadBase, self).__init__()
        self.is_stochanic = True

    @abstractmethod
    def forward_stochanic(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def forward_determistic(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        if self.is_stochanic:
            return self.forward_stochanic(*args, **kwargs)
        else:
            return self.forward_determistic(*args, **kwargs)


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
