def is_valid_layer(layer):
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
