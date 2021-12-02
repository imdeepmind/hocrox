"""Padding layer for Hocrox."""
import cv2

from hocrox.utils import Layer


class Padding(Layer):
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

        self.__top = top
        self.__bottom = bottom
        self.__right = right
        self.__left = left
        self.__color = color

        super().__init__(
            name,
            "padding",
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
            f"Top: {self.__top}, Bottom: {self.__bottom}, Left: {self.__left}, Right: {self.__right}",
        )

    def _apply_layer(self, images, name=None):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        transformed_images = []

        for image in images:
            transformed_images.append(
                cv2.copyMakeBorder(
                    image, self.__top, self.__bottom, self.__left, self.__right, cv2.BORDER_CONSTANT, value=self.__color
                )
            )

        return transformed_images
