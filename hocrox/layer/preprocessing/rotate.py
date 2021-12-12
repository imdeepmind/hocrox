"""Rotate layer for Hocrox."""
import cv2
import numpy as np

from hocrox.utils import Layer


class Rotate(Layer):
    """Grayscale layer grayscaled an image.

    Here is an example code to use the Crop layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing import Grayscale
    from hocrox.layer import Hocrox

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(Grayscale())

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    @staticmethod
    def __rotate_image(image, angle):
        """Rotate an image to certain angle.

        Args:
            image (ndarray): Image to rorate
            angle (float): Angle to rotate

        Returns:
            ndarray: New rotated image.
        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def __init__(self, angle, name=None):
        """Init method for the Rotate layer.

        Args:
            angle (float): Angle to ratate the image.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the angle is not valid.
        """
        if not (isinstance(angle, int) or isinstance(angle, float)):
            raise ValueError(f"The value {angle} for the argument angle is not valid")

        self.__angle = angle

        super().__init__(
            name,
            "rotate",
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
                "read",
                "rescale",
            ],
            f"Angle: {self.__angle}",
        )

    def _apply_layer(self, images, name=None):
        """Apply the transformation method to change the layer.

        Args:
            images (list[ndarray]): List of images to transform.
            name (str, optional): Name of the image series, used for saving the images. Defaults to None.

        Returns:
            list[ndarray]: Return the transform images
        """
        transformed_images = []

        for image in images:
            transformed_images.append(self.__rotate_image(image, self.__angle))

        return transformed_images
