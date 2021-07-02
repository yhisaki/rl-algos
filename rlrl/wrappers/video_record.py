# from typing import Optional
from typing import List
import numpy as np
from gym.core import Wrapper, Env


class NumpyArrayMonitor(Wrapper):
    def __init__(self, env: Env, enable_pyvirtualdisplay=False) -> None:
        super().__init__(env)
        self.frames: List[np.ndarray] = []
        self.is_recording = False
        self.enable_pyvirtualdisplay = enable_pyvirtualdisplay
        if enable_pyvirtualdisplay:
            from pyvirtualdisplay import Display

            self.d = Display(visible=False, backend="xvfb")

    def reset(self, start_record=False, **kwargs):
        self.frames.clear()
        if start_record:
            self.start_recording()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.is_recording:
            self._capture_frame()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def _capture_frame(self):
        frame = self.env.render(mode="rgb_array").copy()
        # frame shape is (height, width, rgb)
        self.frames.append(frame.transpose(2, 0, 1))  # (rgb, height, width)

    def get_frames_array(self):
        return np.array(self.frames)

    def start_recording(self):
        self.is_recording = True
        if self.enable_pyvirtualdisplay:
            self.d.start()

    def is_frames_empty(self):
        return len(self.frames) == 0

    def save_to_file(self, filename, fps=60):
        import cv2

        frameSize = self.frames[0].shape[1:3]
        print(frameSize)
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"DIVX"), fps, frameSize)
        for frame in self.frames:
            f = frame.transpose(1, 2, 0)
            out.write(f)
        out.release()
