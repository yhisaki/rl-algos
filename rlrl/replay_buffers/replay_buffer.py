import collections
import pickle
from typing import Optional

import torch

from rlrl.collections.random_access_queue import RandomAccessQueue
from rlrl import replay_buffer


class ReplayBuffer(replay_buffer.AbstractReplayBuffer):
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
    self.capacity = capacity
    self.memory = RandomAccessQueue(maxlen=capacity)

  def append(
      self,
      transition
  ):
    self.memory.append(transition)

  def sample(self, num_experiences, **kwargs):
    assert len(self.memory) >= num_experiences
    s = self.memory.sample(num_experiences)
    
    return s

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