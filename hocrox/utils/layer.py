class Layer:
    def __init__(self, name, type, supported_parent_layer, parameter_str, bypass_validation=False):
        if not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        if not isinstance(type, str):
            raise ValueError(f"The value {type} for the argument type is not valid")

        if not isinstance(supported_parent_layer, list):
            raise ValueError(f"The value {supported_parent_layer} for the argument supported_parent_layer is not valid")

        if not isinstance(parameter_str, str):
            raise ValueError(f"The value {parameter_str} for the argument parameter_str is not valid")

        if not isinstance(bypass_validation, bool):
            raise ValueError(f"The value {bypass_validation} for the argument bypass_validation is not valid")

        self.__name = name
        self.__type = type
        self.__supported_parent_layer = supported_parent_layer
        self.__parameter_str = parameter_str
        self.__bypass_validation = bypass_validation

    def get_description(self):
        return (f"{self.__name}({self.type})", self.__parameter_str)

    def get_name(self):
        return self.__name

    def get_type(self):
        return self.__type

    def get_supported_parent_layer(self):
        return self.__supported_parent_layer

    def get_bypass_validation(self):
        return self.__bypass_validation
