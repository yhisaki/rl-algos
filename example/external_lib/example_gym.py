import gym
from rlrl.wrappers import NormalizeActionSpace, CastObservationToFloat32, CastRewardToFloat32

if __name__ == "__main__":
    env = gym.make("Swimmer-v3")
    env = NormalizeActionSpace(CastRewardToFloat32(CastObservationToFloat32(env)))
    env.action_space.seed(0)
    env.seed(0)
    step = 0
    for i in range(10):
        done = False
        reward_sum = 0
        state = env.reset()
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            step += 1
            reward_sum += reward
        print(f"epi : {i}, reward_sum : {reward_sum}, step : {step}")
