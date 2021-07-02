from typing import Dict, List


def transpose_list_dict(lst: List[Dict]) -> Dict:
    """covert list of dictionary to dictionary of list

    lst = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
    ]
    to
    {'a': [1, 1, 1, 1], 'b': [2, 2, 2, 2], 'c': [3, 3, 3, 3]}


    Args:
        lst (List[Dict]): list of dictionary

    Returns:
        Dict: dictionary of list
    """
    dct = {}
    first = lst[0]

    for first_key in first.keys():
        dct[first_key] = []

    for e in lst:
        for key in dct.keys():
            dct[key].append(e[key])

    return dct
