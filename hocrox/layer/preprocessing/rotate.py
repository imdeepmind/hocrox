"""Rotate layer for Hocrox."""
import cv2
import numpy as np


class Rotate:
    """Rotate layer for Hocrox."""

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

    def __init__(self, angle, name=None):
        """Init method for the Rotate layer.

        :param angle: angle to rotate
        :type angle: float
        :param name: name of the layer
        :type name: str
        """
        if not (isinstance(angle, int) or isinstance(angle, float)):
            raise ValueError(f"The value {angle} for the argument angle is not valid")

        if name and not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        self.__angle = angle
        self.__name = name if name else "Rotate Layer"

        self.type = "rotate"
        self.supported_parent_layer = ["resize", "greyscale", "rotate", "crop", "padding", "save"]
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
            transformed_images.append(self.__rotate_image(image, self.__angle))

        return transformed_images

    def get_description(self):
        """Return layers details for the model to generate summary.

        :return: layer details
        :rtype: str
        """
        return (f"{self.__name}({self.type})", f"Angle: {self.__angle}")
