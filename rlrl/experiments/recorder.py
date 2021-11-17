import logging
from typing import List

import gym
import numpy as np
from gym import Env
from gym.wrappers.pixel_observation import PixelObservationWrapper


def record_videos_from_actor(
    env: gym.Env,
    actor,
    num_videos=1,
    frame_skip=1,
    pixel=False,
    dir=None,
    logger: logging.Logger = logging.getLogger(__name__),
):
    if pixel:
        videos: List[np.ndarray] = []
        env.reset()
        env = PixelObservationWrapper(env, pixels_only=False)

        for i in range(num_videos):
            video = []
            state_and_pixels = env.reset()
            video.append(state_and_pixels["pixels"].transpose(2, 0, 1))
            reward_sum = 0
            step = 0
            while True:
                action = actor(state_and_pixels["state"])
                state_and_pixels, reward, done, info = env.step(action)
                if step % frame_skip == 0:
                    video.append(state_and_pixels["pixels"].transpose(2, 0, 1))
                reward_sum += reward
                step += 1
                if done:
                    break

            logger.info(
                f"Recording video {i+1}/{num_videos}, reward_sum={reward_sum}, step = {step}"
            )
            videos.append(np.array(video))

        return videos
    elif isinstance(dir, str):
        raise NotImplementedError()  # TODO


class Recoder(object):
    def __init__(
        self,
        env: Env,
        record_interval: int = 10e4,
        use_pyvirtualdisplay: bool = False,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        super().__init__()
        self.env = env

        self.record_interval = record_interval
        self.pre_record_step = 0

        if use_pyvirtualdisplay:
            try:
                from pyvirtualdisplay import Display

                self.d = Display(visible=False, backend="xvfb")
            except ModuleNotFoundError:
                self.logger.warning("Can not import pyvirtualdisplay")

        self.logger = logger

    def record_videos(self, actor, num_videos=1, pixel: bool = True, dir: str = None):
        return record_videos_from_actor(
            env=self.env,
            actor=actor,
            num_videos=num_videos,
            pixel=pixel,
            dir=dir,
            logger=self.logger,
        )

    def record_videos_if_necessary(
        self, total_steps: int, actor, num_videos=1, pixel: bool = True, dir: str = None
    ):
        videos = []
        if self.record_interval < 0:
            return videos
        n = total_steps // self.record_interval
        if (self.pre_record_step < n * self.record_interval) and (
            n * self.record_interval <= total_steps
        ):
            videos = self.record_videos(
                actor=actor,
                num_videos=num_videos,
                pixel=pixel,
                dir=dir,
            )
        self.pre_record_step = total_steps
        return videos
