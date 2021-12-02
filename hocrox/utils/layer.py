"""Layer class is used to make layers for Hocrox."""


class Layer:
    """Layer class is used to make layers for Hocrox."""

    def __init__(self, name, type, supported_parent_layer, parameter_str, bypass_validation=False):
        """Init method for the Layer class.

        Arguments:
            name {str} -- Name of the layer
            type {str} -- Type of the layer
            supported_parent_layer {list} -- List of supported parent layers
            parameter_str {str} -- The parameter string for printing the model summary

        Keyword Arguments:
            bypass_validation {bool} -- Flag to bypass layer validation, useful when making
            custom layers (default: {False})

        Raises:
            ValueError: If the name is not valid
            ValueError: If the type is not valid
            ValueError: If the supported_parent_layer is not valid
            ValueError: If the parameter_str is not valid
            ValueError: If the bypass_validation is not valid
        """
        if name and not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        if not isinstance(type, str):
            raise ValueError(f"The value {type} for the argument type is not valid")

        if not isinstance(supported_parent_layer, list):
            raise ValueError(f"The value {supported_parent_layer} for the argument supported_parent_layer is not valid")

        if not isinstance(parameter_str, str):
            raise ValueError(f"The value {parameter_str} for the argument parameter_str is not valid")

        if not isinstance(bypass_validation, bool):
            raise ValueError(f"The value {bypass_validation} for the argument bypass_validation is not valid")

        self.__name = name or f"{type.capitalize().replace('_', ' ')} Layer"
        self.__type = type
        self.__supported_parent_layer = supported_parent_layer
        self.__parameter_str = parameter_str
        self.__bypass_validation = bypass_validation

    def _get_description(self):
        """Return the description tuple of the layer, used in printing the model summary.

        Returns:
            tuple -- Description of the layer
        """
        return (f"{self.__name}({self.__type})", self.__parameter_str)

    def _is_valid_child(self, previous_layer_type):
        """Check if based on the previous layer type, the current is valid or not.

        Arguments:
            previous_layer_type {str} -- Type of the previous layer

        Returns:
            bool -- True if the layer is valid, else False
        """
        if self.__bypass_validation:
            return True

        return previous_layer_type in self.__supported_parent_layer

    def _get_name(self):
        """Return the name of the layer.

        Returns:
            str -- Name of the layer
        """
        return self.__name

    def _get_type(self):
        """Return the type of the layer.

        Returns:
            str -- Type of the layer
        """
        return self.__type
