"""Crop layer for Hocrox."""
from hocrox.utils import Layer


class Crop(Layer):
    """Crop layer crops an image.

    Here is an example code to use the Crop layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing import Crop
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(Crop(x=10, y=10, w=100, h=100))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, x, y, w, h, name=None):
        """Init method for the crop layer.

        Args:
            x (int): X coordinate for the crop.
            y (int): Y coordinate for the crop.
            w (int): Width of the cropped image.
            h (int): Height of the cropped image.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the parameter x is not valid
            ValueError: If the parameter y is not valid
            ValueError: If the parameter w is not valid
            ValueError: If the parameter h is not valid
        """
        if x and not isinstance(x, int):
            raise ValueError(f"The value {x} for the argument x is not valid")

        if y and not isinstance(y, int):
            raise ValueError(f"The value {y} for the argument y is not valid")

        if w and not isinstance(w, int):
            raise ValueError(f"The value {w} for the argument w is not valid")

        if h and not isinstance(h, int):
            raise ValueError(f"The value {h} for the argument h is not valid")

        self.__x = x
        self.__y = y
        self.__w = w
        self.__h = h

        super().__init__(
            name,
            "crop",
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
            f"X: {self.__x}, Y: {self.__y}, W: {self.__w}, H: {self.__h}",
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
            transformed_images.append(image[self.__x : self.__x + self.__w, self.__y : self.__y + self.__h])

        return transformed_images
