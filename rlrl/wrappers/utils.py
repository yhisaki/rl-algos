from typing import Type

import gym


def remove_wrapper(env: gym.Env, removed_wrapper_class: Type):
    if not hasattr(env, "env"):
        return env
    if type(env) is removed_wrapper_class:
        return env.env
    else:
        env.env = remove_wrapper(env.env, removed_wrapper_class)
        return env


def replace_wrapper(
    env: gym.Env, old_wrapper: Type, new_wrapper: Type, *args, **new_wrapper_kwargs
):
    if not hasattr(env, "env"):
        return env
    if type(env) is old_wrapper:
        return new_wrapper(env.env, *args, **new_wrapper_kwargs)
    else:
        env.env = replace_wrapper(env.env, old_wrapper, new_wrapper, *args, **new_wrapper_kwargs)
        return env
