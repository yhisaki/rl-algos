from contextlib import contextmanager

# It was implemented using https://github.com/pfnet/pfrl/blob/master/pfrl/utils/contexts.py
# as a reference.


@contextmanager
def evaluating(*nets):
    """Temporarily switch to evaluation mode."""
    istrains = []
    for net in nets:
        istrains.append(net.training)
    try:
        for net in nets:
            net.eval()
            yield net
    finally:
        for net, istrain in zip(nets, istrains):
            if istrain:
                net.train()
