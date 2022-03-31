from contextlib import contextmanager

from torch import nn

"""
It was implemented using https://github.com/pfm/pfrl/blob/master/pfrl/utils/contexts.py
as a reference.
"""


@contextmanager
def evaluating(*ms: nn.Module):
    """Temporarily switch to evaluation mode."""
    istrains = [m.training for m in ms]
    try:
        for m in ms:
            m.eval()
        yield ms
    finally:
        for m, istrain in zip(ms, istrains):
            if istrain:
                m.train()
