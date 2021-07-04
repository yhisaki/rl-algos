import gym

from rlrl.replay_buffers import TorchTensorBatch, ReplayBuffer
from rlrl.wrappers import CastObservationToFloat32


if __name__ == "__main__":
    env = gym.make("Swimmer-v2")
    env = CastObservationToFloat32(env)
    done = False

    buffer = ReplayBuffer(1e4)
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        buffer.append(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            terminal=done,
        )
        state = next_state
    s = buffer.sample(3)
    batch = TorchTensorBatch(**s)
    batch.to("cuda")
    print(batch.reward)
