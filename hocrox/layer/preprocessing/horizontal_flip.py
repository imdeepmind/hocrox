"""HorizontalFlip layer for Hocrox."""
import cv2

from hocrox.utils import Layer


class HorizontalFlip(Layer):
    """HorizontalFlip layer horizontally flips an image.

    Here is an example code to use the Crop layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing import HorizontalFlip

    # Initializing the model
    model = Model("./img")

    # Adding model layers
    model.add(HorizontalFlip())

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, name=None):
        """Init method for horizontal flip layer.

        Args:
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        """
        super().__init__(
            name,
            "horizontal_flip",
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
            "-",
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
            transformed_images.append(cv2.flip(image, 1))

        return transformed_images
