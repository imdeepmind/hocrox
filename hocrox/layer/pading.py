"""Padding layer for Hocrox."""
import cv2


class Padding:
    """Padding layer for Hocrox."""

    def __init__(self, top, bottom, left, right, color=[255, 255, 255], name=None):
        """Init method for the Padding layer.

        :param x: x coordinate
        :type x: int
        :param y: y coordinate
        :type y: int
        :param w: w coordinate
        :type w: int
        :param h: h coordinate
        :type h: int
        :param h: h coordinate
        :type h: int

        :param name: name of the layer
        :type name: str
        """
        if top and not isinstance(top, int):
            raise ValueError(f"The value {top} for the argument top is not valid")

        if bottom and not isinstance(bottom, int):
            raise ValueError(f"The value {bottom} for the argument bottom is not valid")

        if left and not isinstance(left, int):
            raise ValueError(f"The value {left} for the argument left is not valid")

        if right and not isinstance(right, int):
            raise ValueError(f"The value {right} for the argument h is not valid")

        if name and not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        if not isinstance(color, list) or len(color) != 3:
            raise ValueError(f"The value {color} for the argument color is not valid")

        self.__top = top
        self.__bottom = bottom
        self.__right = right
        self.__left = left
        self.__color = color
        self.__name = name if name else "Padding Layer"

        self.type = "padding"
        self.supported_parent_layer = ["resize", "greyscale", "rotate", "crop", "padding", "save"]
        self.bypass_validation = False

    def apply_layer(self, img, name=None):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        return cv2.copyMakeBorder(
            img, self.__top, self.__bottom, self.__left, self.__right, cv2.BORDER_CONSTANT, value=self.__color
        )

    def get_description(self):
        """Return layers details for the model to generate summary.

        :return: layer details
        :rtype: str
        """
        return (
            f"{self.__name}({self.type})",
            f"Top: {self.__top}, Bottom: {self.__bottom}, Left: {self.__left}, Right: {self.__right}",
        )
