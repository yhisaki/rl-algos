from typing import Dict, List


def transpose_list_dict(lst: List[Dict], use_all_keys=True) -> Dict:
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
    keys = lst[0].keys()
    for dct in lst:
        if use_all_keys:
            keys |= dct.keys()
        else:
            keys &= dct.keys()
    return {key: [dct[key] if key in dct else None for dct in lst] for key in keys}
