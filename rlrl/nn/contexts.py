from contextlib import contextmanager
from rlrl.utils.get_module_device import get_module_device

# It was implemented using https://github.com/pfm/pfrl/blob/master/pfrl/utils/contexts.py
# as a reference.


@contextmanager
def evaluating(*ms):
    """Temporarily switch to evaluation mode."""
    istrains = [m.training for m in ms]
    try:
        for m in ms:
            m.eval()
            yield m
    finally:
        for m, istrain in zip(ms, istrains):
            if istrain:
                m.train()


@contextmanager
def device_placement(dev, *ms):
    """Temporarily place m on the specified device."""
    pre_devs = [get_module_device(m) for m in ms]
    try:
        for m in ms:
            m.to(dev)
            yield m
    finally:
        for m, dev in zip(ms, pre_devs):
            m.to(dev)
