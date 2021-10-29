from rlrl.experiments import record_videos_from_actor
import logging
import wandb
import gym
from pyvirtualdisplay import Display

if __name__ == "__main__":
    with Display(visible=False, backend="xvfb"):
        logging.basicConfig(level=logging.INFO)
        wandb.init(project="example_rlrl")
        with gym.make("BipedalWalker-v3") as env:

            def actor(state):
                return env.action_space.sample()

            videos = record_videos_from_actor(env, actor, num_videos=1, pixel=True)

            wandb.log({"video": wandb.Video(videos[0], fps=60)})
