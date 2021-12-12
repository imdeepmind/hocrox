"""Resscale layer for Hocrox."""
from hocrox.utils import Layer


class Resscale(Layer):
    """Resscale layer rescales an image.

    Here is an example code to use the Resscale layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing import Resscale
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(Resscale(rescale=1.0 / 255.0))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, rescale=1.0 / 255.0, name=None):
        """Init method for Resize layer.

        Args:
            rescale (float, optional): Rescale factor for rescaling image, Defaults to 1.0 / 255.0.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the rescale parameter is not valid
        """
        if not isinstance(rescale, float):
            raise ValueError(f"The value {rescale} for the argument rescale is not valid")

        self.__rescale = rescale

        super().__init__(
            name,
            "rescale",
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
            f"Rescale: {self.__rescale}",
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
            transformed_images.append(image * self.__rescale)

        return transformed_images
