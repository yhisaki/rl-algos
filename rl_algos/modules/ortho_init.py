from torch import nn


def ortho_init(layer: nn.Linear, gain: float):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer
