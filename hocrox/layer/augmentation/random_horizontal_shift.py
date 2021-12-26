"""RandomHorizontalShift layer for Hocrox."""
import random
import cv2

from hocrox.utils import Layer


class RandomHorizontalShift(Layer):
    """RandomHorizontalShift layer randomly zooms the image.

    Here is an example code to use the RandomHorizontalShift layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation import RandomHorizontalShift
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(RandomHorizontalShift(low=1, high=5, number_of_outputs=1))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, ratio=0.7, number_of_outputs=1, name=None):
        """Init method for the RandomHorizontalShift layer.

        Args:
            ratio (float, optional): Starting range of the brightness. Defaults to 0.5.
            number_of_outputs (int, optional): Number of images to output. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ratioError: If the low parameter is not valid
            ValueError: If the high parameter is not valid
            ValueError: If the number_of_images parameter is not valid
        """
        if not (isinstance(ratio, float)):
            raise ValueError(f"The value {ratio} for the argument ratio is not valid")

        if isinstance(number_of_outputs, int) and number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        super().__init__(
            name,
            "random_channel_shift",
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
            ],
            f"Ratio:{ratio}, Number of Outputs: {number_of_outputs}",
        )

        self.__number_of_outputs = number_of_outputs
        self.__ratio = ratio

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
                transformed_images.append(self.__horizontal_shift(image, self.__ratio))

        return transformed_images

    @staticmethod
    def __horizontal_shift(img, ratio):
        """Apply horizontal_shift function to the image.

        Args:
            img (ndarray): Image to change the brightness
            ratio (float): High range of the brightness

        Returns:
            ndarray: Updated image
        """
        ratio = random.uniform(-ratio, ratio)

        h, w = img.shape[:2]
        to_shift = w * ratio
        if ratio > 0:
            img = img[:, : int(w - to_shift), :]
        if ratio < 0:
            img = img[:, int(-1 * to_shift) :, :]

        img = img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)

        return img
