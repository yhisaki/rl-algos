from rlrl.experiments import record_videos_from_actor
import logging
import wandb
import gym
from pyvirtualdisplay import Display


def main(env_id: str):
    logging.basicConfig(level=logging.INFO)
    wandb.init(project="example_rlrl")
    with gym.make(env_id) as env:

        def actor(state):
            return env.action_space.sample()

        videos = record_videos_from_actor(env, actor, num_videos=1, pixel=True, frame_skip=2)

        wandb.log({"video": wandb.Video(videos[0], fps=30, format="mp4")})


if __name__ == "__main__":
    with Display(visible=False, backend="xvfb"):
        main("Swimmer-v3")
