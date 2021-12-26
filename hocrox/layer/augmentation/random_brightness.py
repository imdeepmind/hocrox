"""RandomBrightness layer for Hocrox."""
import cv2
import random
import numpy as np

from hocrox.utils import Layer


class RandomBrightness(Layer):
    """RandomBrightness layer randomly changes the brightness of the image based on the provided low and high.

    Here is an example code to use the RandomBrightness layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation import RandomBrightness
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(RandomBrightness(low=0.5, high=3.0, number_of_outputs=1))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, low=0.5, high=3.0, number_of_outputs=1, name=None):
        """Init method for the RandomBrightness layer.

        Args:
            low (float, optional): Starting range of the brightness. Defaults to 0.5.
            end (float, optional): Ending range of the brightness. Defaults to 3.0.
            number_of_outputs (int, optional): Number of images to output. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the low parameter is not valid
            ValueError: If the high parameter is not valid
            ValueError: If the number_of_images parameter is not valid
        """
        if not (isinstance(low, float)):
            raise ValueError(f"The value {low} for the argument low is not valid")

        if not (isinstance(high, float)):
            raise ValueError(f"The value {high} for the argument high is not valid")

        if not isinstance(number_of_outputs, int) or number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        super().__init__(
            name,
            "random_brightness",
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
                "random_zoom",
                "random_brightness",
                "random_channel_shift",
                "random_horizontal_shift",
                "random_vertical_shift",
            ],
            f"Low: {low}, High:{high}, Number of Outputs: {number_of_outputs}",
        )

        self.__number_of_outputs = number_of_outputs
        self.__low = low
        self.__high = high

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
            for _ in range(self.__number_of_outputs):
                transformed_images.append(self.__brightness(image, self.__low, self.__high))

        return transformed_images

    @staticmethod
    def __brightness(img, low, high):
        """Apply brightness function to the image.

        Args:
            img (ndarray): Image to change the brightness
            low (float): Low range of the brightness
            high (float): High range of the brightness

        Returns:
            ndarray: Updated image
        """
        value = random.uniform(low, high)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)

        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img
