"""Crop layer for Hocrox."""


class Crop:
    """Crop layer for Hocrox."""

    def __init__(self, x, y, w, h, name=None):
        """Init method for the Crop layer.

        :param x: x coordinate
        :type x: int
        :param y: y coordinate
        :type y: int
        :param w: w coordinate
        :type w: int
        :param h: h coordinate
        :type h: int
        :param name: name of the layer
        :type name: str
        """
        if x and not isinstance(x, int):
            raise ValueError(f"The value {x} for the argument x is not valid")

        if y and not isinstance(y, int):
            raise ValueError(f"The value {y} for the argument y is not valid")

        if w and not isinstance(w, int):
            raise ValueError(f"The value {w} for the argument w is not valid")

        if h and not isinstance(h, int):
            raise ValueError(f"The value {h} for the argument h is not valid")

        if name and not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        self.__x = x
        self.__y = y
        self.__w = w
        self.__h = h
        self.__name = name if name else "Crop Layer"

        self.type = "crop"
        self.supported_parent_layer = ["resize", "greyscale", "rotate", "crop", "padding", "save", "horizontal_flip"]
        self.bypass_validation = False

    def apply_layer(self, images, name=None):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        transformed_images = []

        for image in images:
            transformed_images.append(image[self.__x : self.__x + self.__w, self.__y : self.__y + self.__h])

        return transformed_images

    def get_description(self):
        """Return layers details for the model to generate summary.

        :return: layer details
        :rtype: str
        """
        return (f"{self.__name}({self.type})", f"X: {self.__x}, Y: {self.__y}, W: {self.__w}, H: {self.__h}")
