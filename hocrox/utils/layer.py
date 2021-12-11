"""Layer class is used to make layers for Hocrox."""


class Layer:
    """Layer class is used to make layers for Hocrox.

    Here is an example code for making a new custom CropCustom Hocrox layer using this Layer class as a base class.

    ```python
    from hocrox.utils import Layer

    class CropCustom(Layer):
        def __init__(self, x, y, w, h, name=None):
            if x and not isinstance(x, int):
                raise ValueError(f"The value {x} for the argument x is not valid")

            if y and not isinstance(y, int):
                raise ValueError(f"The value {y} for the argument y is not valid")

            if w and not isinstance(w, int):
                raise ValueError(f"The value {w} for the argument w is not valid")

            if h and not isinstance(h, int):
                raise ValueError(f"The value {h} for the argument h is not valid")

            self.__x = x
            self.__y = y
            self.__w = w
            self.__h = h

            super().__init__(
                name,
                "crop",
                [
                    "resize",
                    "greyscale",
                    "rotate",
                    "crop",
                    "padding",
                    "save",
                    "horizontal_flip",
                    "vertical_flip",
                    "random_rotate",
                    "random_flip",
                ],
                f"X: {self.__x}, Y: {self.__y}, W: {self.__w}, H: {self.__h}",
            )

        # This method below receives a list of images and name of the image, transforms the images, and finally
        # returns the transformed image
        def _apply_layer(self, images, name=None):
            transformed_images = []

            for image in images:
                transformed_images.append(image[self.__x : self.__x + self.__w, self.__y : self.__y + self.__h])

            return transformed_images
    ```
    """

    def __init__(self, name, type, supported_parent_layer, parameter_str, bypass_validation=False):
        """Init method for Layer class.

        Args:
            name (str): Name of the layer.
            type (str): Type of the layer
            supported_parent_layer (list): List of layers that the current layers support as a parent.
            parameter_str (str): Parameter string used for model summary generation.
            bypass_validation (bool, optional): Flag to bypass validation flags in model.add() function. Used heavy
                when making new custom layers. Defaults to False.

        Raises:
            ValueError: If the name parameter is not valid.
            ValueError: If the type parameter is not valid.
            ValueError: If the supported_parent_layer parameter is not valid.
            ValueError: If the parameter_str parameter is not valid.
            ValueError: If the bypass_validation parameter is not valid.
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
        """Return the description string of the layer.

        Used for generating model summary.

        Returns:
            tuple: Description of the layer.
        """
        return (f"{self.__name}({self.__type})", self.__parameter_str)

    def _is_valid_child(self, previous_layer_type):
        """Check if based on the previous layer type, the current is valid or not.

        Arguments:
            previous_layer_type (str): Type of the previous layer

        Returns:
            bool: True if the layer is valid, else False
        """
        if self.__bypass_validation:
            return True

        return previous_layer_type in self.__supported_parent_layer

    def _get_name(self):
        """Return the name of the layer.

        Returns:
            str: Name of the layer
        """
        return self.__name

    def _get_type(self):
        """Return the type of the layer.

        Returns:
            str: Type of the layer
        """
        return self.__type
