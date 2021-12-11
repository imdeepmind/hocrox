"""is_valid_layer method is used to check if a layer is valid or not."""


def is_valid_layer(layer):
    """Check if the layer is valid or not.

    This function is used to check if the layer is valid or not. It should be used when building custom layers.

    Args:
        layer (class): Custom layer class.

    Returns:
        bool: True if the layer is valid, else False.
    """
    if not hasattr(layer, "_get_description"):
        return False

    if not hasattr(layer, "_get_name"):
        return False

    if not hasattr(layer, "_get_type"):
        return False

    if not hasattr(layer, "_is_valid_child"):
        return False

    if not hasattr(layer, "_apply_layer"):
        return False

    return True
