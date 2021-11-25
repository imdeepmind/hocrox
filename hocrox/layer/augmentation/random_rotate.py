"""RandomRotate layer for Hocrox."""
import cv2
import random
import numpy as np

from hocrox.utils import Layer


class RandomRotate(Layer):
    """RandomRotate layer for Hocrox."""

    @staticmethod
    def __rotate_image(image, angle):
        """Rotate the image to specific angle.

        :param image: image array
        :type image: ndarray
        :param angle: angle to rotate
        :type angle: float
        :return: rotated image
        :rtype: ndarray
        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return result

    def __init__(self, start_angle, end_angle, number_of_outputs=1, name=None):
        """Init method for the Rotate layer.

        :param angle: start of the angle range
        :type angle: float
        :param angle: end of the angle range
        :type angle: float
        :param number_of_outputs: number of images to output
        :type angle: int
        :param name: name of the layer
        :type name: str
        """
        if not (isinstance(start_angle, int) or isinstance(start_angle, float)):
            raise ValueError(f"The value {start_angle} for the argument start_angle is not valid")

        if not (isinstance(end_angle, int) or isinstance(end_angle, float)):
            raise ValueError(f"The value {end_angle} for the argument end_angle is not valid")

        if start_angle > end_angle:
            raise ValueError(f"The value {start_angle} for the argument start_angle is not valid")

        if isinstance(number_of_outputs, int) and number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        self.__start_angle = start_angle
        self.__end_angle = end_angle
        self.__number_of_outputs = number_of_outputs

        super().__init__(
            name,
            "random_rotate",
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
                angle = random.uniform(self.__start_angle, self.__end_angle)
                transformed_images.append(self.__rotate_image(image, angle))

        return transformed_images
