from typing import Any, Tuple
import torch
from gym.core import Env
import copy  # noqa
from multiprocessing import Pool  # noqa


def _eval_onece(args: Tuple[Any, Env]):
    policy, env = args
    state = env.reset()
    reward_sum = 0
    i = 0
    done = False
    with torch.no_grad():
        while not done:
            action = policy(torch.tensor(state, dtype=torch.float32))
            state, reward, done, _ = env.step(action.cpu().numpy())
            _, reward, done, _ = env.step(action)
            reward_sum += reward
            i += 1
    return reward_sum, env.seed()


def eval_determistic_policy(policy, env: Env, n_times: int = 1):
    rewards_sum_list = [_eval_onece((policy, env)) for _ in range(n_times)]
    return sum(rewards_sum_list)


# def eval_determistic_policy(policy, env: Env, n_times: int = 1):
#     pool = Pool(n_times)
#     args = ((copy.deepcopy(policy), copy.deepcopy(env)) for _ in range(n_times))
#     # with device_placement("cpu", policy):
#     rewards_sum_list = pool.map_async(_eval_onece, args).get()
#     pool.close()
#     pool.join()
#     return rewards_sum_list


# if __name__ == "__main__":
#     import gym
#     from rlrl.utils.tictoc import tic, toc

#     N = 20

#     env = gym.make("Swimmer-v2")
#     env.seed()

#     def random_policy(x):
#         return torch.tensor(env.action_space.sample())

#     # lst = eval_determistic_policy(random_policy, env, N)
#     # print(lst)
#     tic()
#     lst = eval_determistic_policy(random_policy, env, N)
#     # for _ in range(N):
#     #     _eval_onece((random_policy, env))
#     toc()
#     print(lst)
#     # main()
