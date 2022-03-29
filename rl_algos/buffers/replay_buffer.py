import collections
import pickle
from typing import Optional

from rl_algos.collections.random_access_queue import RandomAccessQueue
from rl_algos.utils.transpose_list_dict import transpose_list_dict

from .abstract_replay_buffer import AbstractReplayBuffer


class ReplayBuffer(AbstractReplayBuffer):
    """Experience Replay Buffer

    As described in
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

    Args:
        capacity (int): capacity in terms of number of transitions
        num_steps (int): Number of timesteps per stored transition
            (for N-step updates)
    """

    # Implements AbstractReplayBuffer.capacity
    capacity: Optional[int] = None

    def __init__(self, capacity: Optional[int] = None):
        self.capacity = int(capacity)
        self.memory = RandomAccessQueue(maxlen=capacity)

    def append(self, state, next_state, action, reward, terminal, reset, **kwargs):
        transition = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminal=terminal,
            reset=reset,
            **kwargs
        )

        self.memory.append(transition)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.memory[idx]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self.memory))
            return transpose_list_dict([self.memory[i] for i in range(start, stop, step)])

    def sample(self, n):
        assert len(self.memory) >= n
        s = self.memory.sample(n)
        return transpose_list_dict(s)

    def __len__(self):
        return len(self.memory)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.memory = pickle.load(f)
        if isinstance(self.memory, collections.deque):
            # Load v0.2
            self.memory = RandomAccessQueue(self.memory, maxlen=self.memory.maxlen)
