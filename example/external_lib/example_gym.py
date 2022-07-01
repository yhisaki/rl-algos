import gym

from rl_algos.wrappers import CastObservationToFloat32, CastRewardToFloat, NormalizeActionSpace
from rl_algos.utils import manual_seed

if __name__ == "__main__":
    env = gym.make("Swimmer-v4", disable_env_checker=True)
    env = NormalizeActionSpace(CastRewardToFloat(CastObservationToFloat32(env)))
    manual_seed(0)
    env.action_space.seed(0)
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
