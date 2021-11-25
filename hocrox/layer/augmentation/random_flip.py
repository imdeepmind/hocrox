"""RandomFlip layer for Hocrox."""
import cv2
import random

from hocrox.utils import Layer


class RandomFlip(Layer):
    """RandomFlip layer for Hocrox."""

    def __init__(self, number_of_outputs=1, name=None):
        """Init method for the RandomFlip layer.

        :param number_of_outputs: number of images to output
        :type angle: int
        :param name: name of the layer
        :type name: str
        """
        if isinstance(number_of_outputs, int) and number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        super().__init__(
            name,
            "random_flip",
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
            f"Number of Outputs: {number_of_outputs}",
        )

        self.__number_of_outputs = number_of_outputs

    def apply_layer(self, images, name=None):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        transformed_images = []

        for image in images:
            for _ in range(self.__number_of_outputs):
                flip = random.randint(0, 1)
                transformed_images.append(cv2.flip(image, flip))

        return transformed_images
