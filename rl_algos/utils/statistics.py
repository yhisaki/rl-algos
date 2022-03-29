import collections
from typing import Dict, Iterable, List

import numpy as np


def mean_or_nan(data: Iterable):
    if len(data) == 0:
        return np.NaN
    else:
        return np.mean(data)


def var_or_nan(data: Iterable):
    if len(data) == 0:
        return np.NaN
    else:
        return np.var(data)


def clear_if_maxlen_is_none(*datum: collections.deque):
    for data in datum:
        if data.maxlen is None:
            data.clear()


class Statistics(object):
    def __init__(self) -> None:
        self._memory: Dict[list] = dict()

    def __call__(self, key, methods=["mean"]) -> List[np.ndarray]:
        if key not in self._memory:
            self._memory[key] = {"data": list(), "methods": methods}
        return self._memory[key]["data"]

    def flush(self):
        stats = {}

        for key, memory in self._memory.items():
            for method in memory["methods"]:
                stats[f"{key}_{method}"] = {
                    "mean": lambda x: np.mean(x, axis=0),
                    "var": lambda x: np.var(x, axis=0),
                    "max": lambda x: np.max(x, axis=0),
                    "min": lambda x: np.min(x, axis=0),
                    "latest": lambda x: x[-1],
                }[method](memory["data"])

        self._memory.clear()

        return stats
