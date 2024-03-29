import logging
from typing import Optional, Union

import numpy as np
from gymnasium.core import Env

_warn_onece_flag = False


def is_state_terminal(
    env: Env,
    step: Union[int, np.ndarray],
    done: Union[bool, np.ndarray],
    info: Optional[dict] = None,
):
    if "is_terminal_state" in info:
        return info["is_terminal_state"]
    if hasattr(env.spec, "max_episode_steps"):
        return done & (step < env.spec.max_episode_steps)
    else:
        global _warn_onece_flag
        if not _warn_onece_flag:
            logging.warning(f"{env} do not have env.spec.max_episode_steps")
            _warn_onece_flag = True
        return done
