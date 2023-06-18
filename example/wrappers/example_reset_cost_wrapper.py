from rl_algos.experiments import TransitionGenerator
from rl_algos.wrappers import ResetCostWrapper, make_env, vectorize_env


def example1():
    env = make_env("Hopper-v4")
    env = ResetCostWrapper(env, reset_cost=float("nan"))
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
    env = ResetCostWrapper(make_env("Swimmer-v4"), reset_cost=100.0)
    print(env)

    def actor(*arg, **kwargs):
        return env.action_space.sample()

    interactions = TransitionGenerator(env, actor, max_step=1005)

    for (
        episode_step,
        state,
        next_state,
        action,
        reward,
        terminated,
        truncated,
        info,
    ) in interactions:
        print(state[0, 0], next_state[0, 0], reward, terminated, truncated)


def example3():
    def _make_env(env_id: str):
        env = make_env(env_id=env_id, disable_env_checker=True, max_episode_steps=None)
        return ResetCostWrapper(
            env,
            reset_cost=float("nan"),
        )

    env = vectorize_env(env_id="Swimmer-v4", num_envs=1, env_fn=_make_env)
    print(env)

    def actor(*arg, **kwargs):
        return env.action_space.sample()

    interactions = TransitionGenerator(env, actor, max_step=1005)

    for (
        episode_step,
        state,
        next_state,
        action,
        reward,
        terminated,
        truncated,
        info,
    ) in interactions:
        print(
            f"{episode_step.sum()} : {state[0, 0]}, {next_state[0, 0]}, "
            f"{reward}, {terminated}, {truncated}"
        )


if __name__ == "__main__":
    example1()
