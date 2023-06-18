import random
from typing import Optional

import gymnasium
import numpy as np
import torch


def manual_seed(
    seed: int = 0,
    torch_seed: Optional[int] = None,
    random_seed: Optional[int] = None,
    np_seed: Optional[int] = None,
):
    if seed is None:
        return
    torch.manual_seed(seed if torch_seed is None else torch_seed)
    random.seed(seed if random_seed is None else random_seed)
    np.random.seed(seed if np_seed is None else np_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gymnasium.Env._np_random, _ = gymnasium.utils.seeding.np_random(seed)
