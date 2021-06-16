import types
from torch import nn
from rlrl.nn.lmbda import Lambda


def build_simple_linear_sequential(
    input_dim: int, output_dim: int, hidden_units: list, hidden_activation, output_activation=None
):
    """build_simple_linear_sequential

    Args:
        input_dim (int): [description]
        output_dim (int): [description]
        hidden_units (list): [description]
        hidden_activation (nn.Module or str): [description]
        output_activation ([type], optional): [description]. Defaults to None.

    Returns:
        Linear Neural Network
    """
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        if isinstance(hidden_activation, str):
            model.append(eval("nn." + hidden_activation)())
        else:
            model.append(hidden_activation())
        units = next_units
    model.append(nn.Linear(units, output_dim))

    if isinstance(output_activation, types.FunctionType):
        model.append(Lambda(output_activation))
    elif output_activation is None:
        pass
    elif issubclass(output_activation, nn.Module):
        model.append(output_activation())

    return nn.Sequential(*model)


if __name__ == "__main__":
    f = build_simple_linear_sequential(10, 3, [256, 256], nn.ReLU, nn.Tanh)
    print(f)
    f2 = build_simple_linear_sequential(10, 3, [256, 256], nn.ReLU, lambda x: 2 * x)
    print(f2)
    f3 = build_simple_linear_sequential(10, 3, [256, 256], "ReLU", lambda x: 2 * x)
    print(f3)
