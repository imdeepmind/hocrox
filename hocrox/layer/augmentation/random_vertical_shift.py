"""RandomVerticalShift layer for Hocrox."""
import random
import cv2

from hocrox.utils import Layer


class RandomVerticalShift(Layer):
    """RandomVerticalShift layer randomly shifts the image vertically.

    Here is an example code to use the RandomVerticalShift layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation import RandomVerticalShift
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(RandomVerticalShift(ratio=0.7, number_of_outputs=1))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, ratio=0.7, number_of_outputs=1, name=None):
        """Init method for the RandomVerticalShift layer.

        Args:
            ratio (float, optional): Ratio is used to define the range of the shift. Defaults to 0.7.
            number_of_outputs (int, optional): Number of images to output. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the ratio parameter is not valid
            ValueError: If the number_of_images parameter is not valid
        """
        if not (isinstance(ratio, float)):
            raise ValueError(f"The value {ratio} for the argument ratio is not valid")

        if not isinstance(number_of_outputs, int) or number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        super().__init__(
            name,
            "random_vertical_shift",
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
                transformed_images.append(self.__vertical_shift(image, self.__ratio))

        return transformed_images

    @staticmethod
    def __vertical_shift(img, ratio):
        """Apply horizontal_shift function to the image.

        Args:
            img (ndarray): Image to change the brightness
            ratio (float): High range of the brightness

        Returns:
            ndarray: Updated image
        """
        ratio = random.uniform(-ratio, ratio)

        h, w = img.shape[:2]
        to_shift = h * ratio

        if ratio > 0:
            img = img[: int(h - to_shift), :, :]
        if ratio < 0:
            img = img[int(-1 * to_shift) :, :, :]

        img = img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)

        return img
