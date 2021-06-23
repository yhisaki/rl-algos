# from typing import Optional
from typing import List
import numpy as np
from gym.core import Wrapper, Env


class NumpyArrayMonitor(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.frames: List[np.ndarray] = []
        self.is_recording = False

    def reset(self, **kwargs):
        self.frames.clear()
        self.is_recording = False
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.is_recording:
            self.capture_frame()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def capture_frame(self):
        frame = self.env.render(mode="rgb_array").copy()
        # frame shape is (height, width, rgb)
        self.frames.append(frame.transpose(2, 0, 1))  # (rgb, height, width)

    def get_frames_array(self):
        return np.array(self.frames)

    def start_recording(self):
        self.is_recording = True

    def save_to_file(self, filename, fps=60):
        import cv2

        frameSize = self.frames[0].shape[1:3]
        print(frameSize)
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"DIVX"), fps, frameSize)
        for frame in self.frames:
            f = frame.transpose(1, 2, 0)
            out.write(f)
        out.release()
