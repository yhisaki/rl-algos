def batch_shaping(batch, val_type, **kwargs):
    """サンプリングしたバッチを整形する．

    Args:
        batch ([type]): [description]
        val_type ([type]): [description]

    Returns:
        [type]: [description]
    """
    Transition = type(batch[0])
    return Transition(*map(lambda l: val_type(list(l), **kwargs), zip(*batch)))
