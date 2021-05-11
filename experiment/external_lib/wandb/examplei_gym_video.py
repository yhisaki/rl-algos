from pyvirtualdisplay import Display, display
import gym
import numpy as np
import wandb
import cv2


display = Display()

display.start()

wandb.init(project="openai-gym-trial", monitor_gym=True)

env = gym.make("CartPole-v0")

for _ in range(3):
  env.reset()
  done = False
  i = 0

  frames = []
  rewards = []

  while not done:
    _, r, done, _ = env.step(env.action_space.sample())
    f = env.render(mode='rgb_array').copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(f,
                f'step: {i}',
                (0, 50),
                font, 
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA)
    frames.append(f.transpose(2, 0, 1))
    i += 1
    print(f"step: {i}")

  frames = np.array(frames)

  wandb.log({
    "video": wandb.Video(frames, fps=60, format="mp4")
  })

env.close()

display.stop()
