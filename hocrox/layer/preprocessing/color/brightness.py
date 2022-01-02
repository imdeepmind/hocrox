"""Brightness layer for Hocrox."""
import cv2
import numpy as np

from hocrox.utils import Layer


class Brightness(Layer):
    """RandomBrightness layer changes the brightness of the image based on the provided level.

    Here is an example code to use the Brightness layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing.color import Brightness
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(Brightness(level=0.5))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, level=0.5, name=None):
        """Init method for the Brightness layer.

        Args:
            level (float, optional): Value of the brightness level. Defaults to 0.5.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the level parameter is not valid
        """
        if not (isinstance(level, float)):
            raise ValueError(f"The value {level} for the argument level is not valid")

        super().__init__(
            name,
            "brightness",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Level: {level}",
        )

        self.__level = level

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
            if image is not None and len(image) != 0:
                transformed_image = self.__brightness(image, self.__level)

                if transformed_image is not None and len(transformed_image) != 0:
                    transformed_images.append(transformed_image)

        return transformed_images

    @staticmethod
    def __brightness(img, value):
        """Apply brightness function to the image.

        Args:
            img (ndarray): Image to change the brightness
            value (float): Level of the brightness

        Returns:
            ndarray: Updated image
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)

        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

        hsv = np.array(hsv, dtype=np.uint8)

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img
