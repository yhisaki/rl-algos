import gym

from rlrl.wrappers import NumpyArrayMonitor

SAVE_TO_WANDB = True
ENVID = "Swimmer-v2"
PATH_TO_VIDEO = "output/video.avi"

if __name__ == "__main__":

    env = gym.make(ENVID)
    env = NumpyArrayMonitor(env)

    state = env.reset(start_record=True)
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

    # save video to file
    if PATH_TO_VIDEO is not None:
        env.save_to_file(PATH_TO_VIDEO, 60)

    # save video to WandB
    if SAVE_TO_WANDB:
        import wandb

        run = wandb.init(project="rlrl_example", name="record_gym_video")
        run.log({"video": wandb.Video(env.get_frames_array(), fps=60, format="mp4")})
