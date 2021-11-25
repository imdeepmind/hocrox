"""Grayscale layer for Hocrox."""
import cv2

from hocrox.utils import Layer


class Grayscale(Layer):
    """Grayscale layer for Hocrox."""

    def __init__(self, name=None):
        """Init method for the Grayscale layer.

        :param name: name of the layer
        :type name: str
        """
        super().__init__(
            name,
            "greyscale",
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
            "-",
        )

    def apply_layer(self, images, name=None):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        transformed_images = []

        for image in images:
            transformed_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        return transformed_images
