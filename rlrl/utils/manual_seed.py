import random

import numpy as np
import torch


def manual_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
