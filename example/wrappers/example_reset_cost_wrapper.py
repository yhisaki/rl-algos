from rlrl.experiments import TransitionGenerator
from rlrl.utils.is_state_terminal import is_state_terminal
from rlrl.wrappers import ResetCostWrapper, make_env, vectorize_env


def example1():
    env = make_env("Hopper-v3")
    env = ResetCostWrapper(env, reset_cost=float("nan"), terminal_step=300)
    print(env)

    step = 0
    env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(info)
        step += 1
        if done:
            break

    print(step)


def example2():
    env = ResetCostWrapper(make_env("Hopper-v3"))
    print(env)

    def actor(*arg, **kwargs):
        return env.action_space.sample()

    interactions = TransitionGenerator(env, actor, max_step=1005)

    for step, state, next_state, action, reward, done, info in interactions:
        print(step, done, state[0][0], next_state[0][0], reward, info["is_terminal_state"])


def example3():
    def _make_env(*env_args, **env_kwargs):
        return ResetCostWrapper(make_env(*env_args, **env_kwargs))

    env = vectorize_env(env_id="Hopper-v3", num_envs=3, seed=1, env_fn=_make_env)
    print(env)

    def actor(*arg, **kwargs):
        return env.action_space.sample()

    interactions = TransitionGenerator(env, actor, max_step=1005)

    for step, state, next_state, action, reward, done, info in interactions:
        print(step, is_state_terminal(env, step, done, info))


if __name__ == "__main__":
    example3()
