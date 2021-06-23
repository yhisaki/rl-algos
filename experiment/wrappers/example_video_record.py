import wandb
import gym
from rlrl.wrappers import NumpyArrayMonitor

if __name__ == "__main__":

    env = gym.make("Swimmer-v2")
    print(env)
    env = NumpyArrayMonitor(env)

    state = env.reset()
    done = False
    env.start_recording()

    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

    # save video to file
    env.save_to_file("output/video.avi", 60)

    # save video to WandB
    wandb.init(project="test")
    wandb.log({"video": wandb.Video(env.get_frames_array(), fps=60, format="mp4")})
