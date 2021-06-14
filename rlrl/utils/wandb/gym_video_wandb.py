import numpy as np
import cv2
import wandb


class GymVideoWandb(object):
    def __init__(self, _env) -> None:
        super().__init__()
        self.env = _env
        self.frames = []

    def capture_frame(self, text: str = None) -> None:
        frame: np.ndarray = self.env.render(mode="rgb_array").copy()
        if text is not None:
            org = (0, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)
            thickness = 2
            line_type = cv2.LINE_AA
            # https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
            cv2.putText(frame, text, org, font, font_scale, color, thickness, line_type)
        self.frames.append(frame.transpose(2, 0, 1))

    def get_video(self, fps: int = 60, fmt: str = "mp4"):
        return wandb.Video(np.array(self.frames), fps=fps, format=fmt)

    def clear(self):
        self.env = None
        self.frames.clear()


if __name__ == "__main__":
    import gym

    env = gym.make("CartPole-v0")
    wandb.init(project="test")
    for _ in range(3):
        env.reset()
        done = False
        vid = GymVideoWandb(env)

        while not done:
            _, r, done, _ = env.step(env.action_space.sample())
            vid.capture_frame()

        wandb.log({"video": vid.get_video()})
