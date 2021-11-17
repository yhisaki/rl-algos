from rlrl.experiments import Recoder
import logging
import wandb
import gym


def main(env_id: str):
    logging.basicConfig(level=logging.INFO)
    wandb.init(project="example_rlrl")
    env = gym.make(env_id)

    def actor(state):
        return env.action_space.sample()

    recorder = Recoder(env)
    videos = recorder.record_videos(actor)
    wandb.log({"video": wandb.Video(videos[0], fps=60, format="mp4")})


if __name__ == "__main__":
    main("Swimmer-v3")
