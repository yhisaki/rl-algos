from textwrap import indent


def _blue(string):
    return "\033[34m" + string + "\033[0m"


def _green(string):
    return "\033[32m" + string + "\033[0m"


def dict2colorized_string(title: str, d: dict):
    """[summary]

    Args:
        title (str): Title of Dict
        d (dict): dict

    Returns:
        string

    Example:
    >>> s = dict2colorized_string(
            "=========== Dictionary ===========",
            {
                "a": 12,
                "b": [3, 4, 5],
                "c": {
                    "a": 12,
                    "b": [3, 4, 5],
                },
            },
        )
    >>> print(s)
    =========== Dictionary ===========
    + a:
      12
    + b:
      [3, 4, 5]
    + c:
        + a:
          12
        + b:
          [3, 4, 5]
    """
    return _blue(title) + "\n" + _dic2str(d)


def _dic2str(d: dict, depth=0):
    string = ""
    for key, value in d.items():
        if isinstance(value, dict):
            string += _green(f"+ {key}:") + "\n" + indent(_dic2str(value, depth + 1), "    ")
        else:
            string += _green(f"+ {key}:") + "\n" + indent(str(value), "  ") + "\n"

    return string
