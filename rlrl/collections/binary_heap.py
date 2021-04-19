import numpy as np
import heapq

class EntropyPrioritizedExperience(object):
  def __init__(self, *args, **kwargs):
    self.maxlen = kwargs.pop("maxlen", None)
    assert self.maxlen is None or self.maxlen >= 0

  # def add(self, )