"""RandomChannelShift layer for Hocrox."""
import random
import numpy as np

from hocrox.utils import Layer


class RandomChannelShift(Layer):
    """RandomChannelShift layer randomly adds some value to the channels in the image.

    Here is an example code to use the RandomChannelShift layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation.color import RandomChannelShift
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(RandomChannelShift(low=1, high=5, probability=1.0, number_of_outputs=1))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, low=1, high=5, probability=1.0, number_of_outputs=1, name=None):
        """Init method for the RandomChannelShift layer.

        Args:
            low (int, optional): Starting range of the brightness. Defaults to 0.5.
            high (int, optional): Ending range of the brightness. Defaults to 3.0.
            probability (float, optional): Probability rate for the layer, if the rate of 0.5 then the layer is applied
                on 50% of the images. Defaults to 1.0.
            number_of_outputs (int, optional): Number of images to output. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the low parameter is not valid
            ValueError: If the high parameter is not valid
            ValueError: If the probability parameter is not valid
            ValueError: If the number_of_images parameter is not valid
        """
        if not (isinstance(low, int)):
            raise ValueError(f"The value {low} for the argument low is not valid")

        if not (isinstance(high, int)):
            raise ValueError(f"The value {high} for the argument high is not valid")

        if not isinstance(probability, float) or probability < 0.0 or probability > 1.0:
            raise ValueError(f"The value {probability} for the argument probability is not valid")

        if not isinstance(number_of_outputs, int) or number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        super().__init__(
            name,
            "random_channel_shift",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Low: {low}, High:{high}, Probability: {probability}, Number of Outputs: {number_of_outputs}",
        )

        self.__number_of_outputs = number_of_outputs
        self.__low = low
        self.__high = high
        self.__probability = probability

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
                should_perform = self._get_probability(self.__probability)

                if image is not None and len(image) != 0:
                    transformed_image = (
                        self.__channel_shift(image, self.__low, self.__high) if should_perform else image
                    )

                    if transformed_image is not None and len(transformed_image) != 0:
                        transformed_images.append(transformed_image)

        return transformed_images

    @staticmethod
    def __channel_shift(img, low, high):
        """Apply channel_shift function to the image.

        Args:
            img (ndarray): Image to change the brightness
            low (float): Low range of the brightness
            high (float): High range of the brightness

        Returns:
            ndarray: Updated image
        """
        value = random.uniform(low, high)

        img = img + value
        img[:, :, :][img[:, :, :] > 255] = 255
        img[:, :, :][img[:, :, :] < 0] = 0

        img = img.astype(np.uint8)

        return img
