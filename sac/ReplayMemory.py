import pandas as pd
import numpy as np
import random
from collections import namedtuple
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mask'))


class ReplayMemory(object):
  def __init__(self, capacity: int):
    self.memData_ = []
    self.capacity_ = capacity
    self.position_: int = 0

  def store(self, state, action, next_state, reward, terminal):
    if len(self.memData_) < self.capacity_:
      self.memData_.append(None)
    self.memData_[int(self.position_)] = Transition(
        state, action, next_state, reward, not terminal)
    self.position_ = (self.position_+1) % self.capacity_

  def sample_transitions(self, batch_size: int):
    batch = random.sample(self.memData_, batch_size)
    return batch

  def sample_batch(self, batch_size: int, batch_type: str = "List", device=None) -> Transition:
    transitions = random.sample(self.memData_, batch_size)
    if batch_type == "List":
      return Transition(*zip(*transitions))
    elif (batch_type == "FloatTensor") & (device is None):
      return Transition(*map(torch.FloatTensor, zip(*transitions)))
    elif (batch_type == "FloatTensor") & (device is not None):
      return Transition(*map(lambda l: torch.FloatTensor(list(l)).to(device), zip(*transitions)))

  def __len__(self):
    return len(self.memData_)
