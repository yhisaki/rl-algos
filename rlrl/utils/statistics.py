import collections
from typing import Iterable

from numpy import NaN, mean, var


def mean_or_nan(data: Iterable):
    if len(data) == 0:
        return NaN
    else:
        return mean(data)


def var_or_nan(data: Iterable):
    if len(data) == 0:
        return NaN
    else:
        return var(data)


def clear_if_maxlen_is_none(*datum: collections.deque):
    for data in datum:
        if data.maxlen is None:
            data.clear()
