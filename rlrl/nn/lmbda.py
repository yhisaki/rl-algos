from torch import nn

# copy from https://github.com/pfnet/pfrl/blob/master/pfrl/nn/lmbda.py


class Lambda(nn.Module):
    """Wraps a callable object to make a `torch.nn.Module`.

    This can be used to add callable objects to `torch.nn.Sequential` or
    `pfrl.nn.RecurrentSequential`, which only accept
    `torch.nn.Module`s.

    Args:
        lambd (callable): Callable object.
    """

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

    def __repr__(self):
        return f"Lambda({self.lambd.__name__})"
