import numpy as np
import random
import torch

_GLOBAL_SEED: int = None


def set_global_seed(env, seed: int):
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.action_space.np_random.seed(seed)
