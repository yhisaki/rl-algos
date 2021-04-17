import torch

_GLOBAL_DEVICE: str = 'cpu'


def set_global_torch_device(dev: str):
  # pylint: disable=global-statement
  global _GLOBAL_DEVICE
  if dev == 'cpu':
    pass
  elif torch.cuda.is_available():
    _GLOBAL_DEVICE = dev
  else:
    IndexError("There is no GPU device")


def get_global_torch_device():
  return _GLOBAL_DEVICE
