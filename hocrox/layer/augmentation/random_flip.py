"""RandomFlip layer for Hocrox."""
import cv2
import random

from hocrox.utils import Layer


class RandomFlip(Layer):
    """RandomFlip layer randomly flips the image vertically or horizontally.

    Here is an example code to use the RandomFlip layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation import RandomFlip

    # Initializing the model
    model = Model("./img")

    # Adding model layers
    model.add(RandomFlip(number_of_outputs=1))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, number_of_outputs=1, name=None):
        """Init method for the RandomFlip layer.

        Args:
            number_of_outputs (int, optional): Number of images to output. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the number_of_images parameter is not valid
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
                flip = random.randint(0, 1)
                transformed_images.append(cv2.flip(image, flip))

        return transformed_images
