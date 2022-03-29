import gym
from pyvirtualdisplay import Display

from rl_algos.wrappers import CastObservationToFloat32

if __name__ == "__main__":
    display = Display(visible=False, backend="xvfb")
    display.start()
    env = gym.make("BipedalWalker-v3")
    env = CastObservationToFloat32(env)

    done = False
    env.reset()
    while not done:
        env.render()
        action = env.action_space.high
        state, reward, done, _ = env.step(action)

    display.stop()
