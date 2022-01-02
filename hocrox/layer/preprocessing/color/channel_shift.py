"""ChannelShift layer for Hocrox."""
import numpy as np

from hocrox.utils import Layer


class ChannelShift(Layer):
    """ChannelShift layer adds a value to the channels in the image.

    Here is an example code to use the ChannelShift layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation.color import ChannelShift
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(ChannelShift(value=1))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, value=1, name=None):
        """Init method for the ChannelShift layer.

        Args:
            value (int, optional): The value by which the channel will be shifted. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the value parameter is not valid
        """
        if not (isinstance(value, int)):
            raise ValueError(f"The value {value} for the argument value is not valid")

        super().__init__(
            name,
            "channel_shift",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Value: {value}",
        )

        self.__value = value

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
                transformed_image = self.__channel_shift(image, self.__value)

                if transformed_image is not None and len(transformed_image) != 0:
                    transformed_images.append(transformed_image)

        return transformed_images

    @staticmethod
    def __channel_shift(img, value):
        """Apply channel_shift function to the image.

        Args:
            img (ndarray): Image to make the change
            value (int): Value by which, the channel will be shifted

        Returns:
            ndarray: Updated image
        """
        img = img + value
        img[:, :, :][img[:, :, :] > 255] = 255
        img[:, :, :][img[:, :, :] < 0] = 0

        img = img.astype(np.uint8)

        return img
