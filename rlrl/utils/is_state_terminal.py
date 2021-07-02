from gym.core import Env


def is_state_terminal(env: Env, step: int, done: bool):
    if hasattr(env.spec, "max_episode_steps"):
        return done if step < env.spec.max_episode_steps else False
    else:
        return done
