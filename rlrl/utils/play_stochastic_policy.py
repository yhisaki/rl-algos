import os
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from rlrl.utils.get_module_device import get_module_device


def play_stochastic_policy(env, policy, path_to_video, title: str, n_times: int = 1):
    _my_makedirs(path_to_video)
    dev = get_module_device(policy)
    if path_to_video[-1] != "/":
        path_to_video = path_to_video + "/"
    for i in range(n_times):
        vid = VideoRecorder(env, path=path_to_video + title + str(i) + ".mp4")
        state: np.ndarry = env.reset()
        reward_sum = 0
        done = False
        while not done:
            vid.capture_frame()
            with torch.no_grad():
                # pylint: disable-msg=not-callable,line-too-long
                action_distrib = policy(torch.tensor(state, dtype=torch.float32, device=dev))
                action = action_distrib.sample()
                action = action.cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = next_state
        vid.close()
        os.remove(path_to_video + title + str(i) + ".meta.json")


def _my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == "__main__":
    import gym

    # pylint: disable=ungrouped-imports
    from rlrl.policies import SquashedGaussianPolicy
    from rlrl.utils import get_env_info
    from pyvirtualdisplay import Display

    d = Display()
    d.start()

    gym_env = gym.make("BipedalWalker-v3")
    env_info = get_env_info(gym_env)
    p = SquashedGaussianPolicy(env_info, 20).to("cuda")

    play_stochastic_policy(gym_env, p, "log/test/popopo", "poyo", 2)
    play_stochastic_policy(gym_env, p, "log/test/popopo2", "poyo", 2)
    d.stop()
