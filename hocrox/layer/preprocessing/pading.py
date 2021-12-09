"""Padding layer for Hocrox."""
import cv2

from hocrox.utils import Layer


class Padding(Layer):
    """Padding layer add a padding to an image.

    Here is an example code to use the Padding layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing import Padding

    # Initializing the model
    model = Model("./img")

    # Adding model layers
    model.add(Padding(top=20, bottom=20, left=20, right=20, color=[255, 255, 255]))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, top, bottom, left, right, color=[255, 255, 255], name=None):
        """Init method for Padding layer.

        Args:
            top (int): Top padding size.
            bottom (int): Bottom padding size.
            left (int): Left padding size.
            right (int): Right padding size.
            color (list, optional): Color of the padding. Defaults to [255, 255, 255].
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the top parameter is not valid.
            ValueError: If the bottom parameter is not valid.
            ValueError: If the left parameter is not valid.
            ValueError: If the right parameter is not valid.
            ValueError: If the color parameter is not valid.
        """
        if top and not isinstance(top, int):
            raise ValueError(f"The value {top} for the argument top is not valid")

        if bottom and not isinstance(bottom, int):
            raise ValueError(f"The value {bottom} for the argument bottom is not valid")

        if left and not isinstance(left, int):
            raise ValueError(f"The value {left} for the argument left is not valid")

        if right and not isinstance(right, int):
            raise ValueError(f"The value {right} for the argument right is not valid")

        if not isinstance(color, list) or len(color) != 3:
            raise ValueError(f"The value {color} for the argument color is not valid")

        self.__top = top
        self.__bottom = bottom
        self.__right = right
        self.__left = left
        self.__color = color

        super().__init__(
            name,
            "padding",
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
            f"Top: {self.__top}, Bottom: {self.__bottom}, Left: {self.__left}, Right: {self.__right}",
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
            transformed_images.append(
                cv2.copyMakeBorder(
                    image, self.__top, self.__bottom, self.__left, self.__right, cv2.BORDER_CONSTANT, value=self.__color
                )
            )

        return transformed_images
