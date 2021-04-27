import types
from torch import nn
from rlrl.nn.lmbda import Lambda


def build_simple_linear_nn(
    input_dim: int,
    output_dim: int,
    hidden_units: list,
    hidden_activation,
    output_activation: nn.Module = None
):
  model = []
  units = input_dim
  for next_units in hidden_units:
    model.append(nn.Linear(units, next_units))
    model.append(hidden_activation())
    units = next_units
  model.append(nn.Linear(units, output_dim))

  if isinstance(output_activation, types.FunctionType):
    model.append(Lambda(output_activation))
  elif issubclass(output_activation, nn.Module):
    model.append(output_activation())

  return nn.Sequential(*model)


if __name__ == '__main__':
  f = build_simple_linear_nn(10, 3, [256, 256], nn.ReLU, nn.Tanh)
  print(f)
  f2 = build_simple_linear_nn(10, 3, [256, 256], nn.ReLU, lambda x: 2 * x)
  print(f2)
