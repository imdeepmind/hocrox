"""is_valid_layer method is used to check if a layer is valid or not."""


def is_valid_layer(layer):
    """is_valid_layer method is used to check if a layer is valid or not.

    Arguments:
        layer {layer} -- The layer class

    Returns:
        bool -- True is the layer is valid, else False
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
