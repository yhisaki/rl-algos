from rlrl.wrappers import ResetCostWrapper, make_env


def main():
    env = make_env("Hopper-v3")
    env = ResetCostWrapper(env, terminal_step=300)
    print(env)

    step = 0
    env.reset()
    while True:
        action = env.action_space.sample()
        _, r, done, _ = env.step(action)
        print(done, r)
        step += 1
        if done:
            break

    print(step)


if __name__ == "__main__":
    main()
