import gymnasium

from rl_algos.wrappers import make_env, register_reset_env
from rl_algos.experiments import TransitionGenerator


def example_simple_reset_env():
    register_reset_env()

    env = gymnasium.make("reset_env/Hopper-v4", reset_cost=float("nan"))

    state, info = env.reset()

    t = 0

    while True:
        next_state, reward, terminated, truncated, info = env.step(env.action_space.sample())

        print(t, ":", state.sum(), next_state.sum(), reward, terminated, truncated)

        if terminated | truncated:
            break
        else:
            state = next_state
            t += 1


def example_with_transition_generator():
    register_reset_env()
    env = make_env("reset_env/Hopper-v4", reset_cost=float("nan"))

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
        print(episode_step, ":", state.sum(), next_state.sum(), reward, terminated, truncated)


def main():
    # example_simple_reset_env()

    example_with_transition_generator()


if __name__ == "__main__":
    main()
