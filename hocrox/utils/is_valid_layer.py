"""is_valid_layer method is used to check if a layer is valid or not."""


def is_valid_layer(layer):
    """Check if the layer is valid or not.

    This function needed to be used when building custom layers to confirm the layer is valid.

    Args:
        layer (class): Custom layer class.

    Returns:
        bool: True if the layer is valid, else False.
    """
    if not hasattr(layer, "get_description"):
        return False

    if not hasattr(layer, "get_name"):
        return False

    if not hasattr(layer, "get_type"):
        return False

    if not hasattr(layer, "get_supported_parent_layer"):
        return False

    if not hasattr(layer, "get_bypass_validation"):
        return False

    if not hasattr(layer, "apply_layer"):
        return False

    return True
