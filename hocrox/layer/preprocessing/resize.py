"""Resize layer for Hocrox."""
import cv2

from hocrox.utils import Layer


class Resize(Layer):
    """Resize layer resize an image to specific dimension.

    Here is an example code to use the Resize layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing import Resize
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(Resize(dim=(200,200), interpolation="INTER_LINEAR"))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, dim, interpolation="INTER_LINEAR", name=None):
        """Init method for Resize layer.

        Args:
            dim (tuple): New dimension for the image
            interpolation (str, optional): Interpolation method for the image. Defaults to "INTER_LINEAR".
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the dim parameter is not valid
            ValueError: If the dim parameter values are less than 0
            ValueError: If the interpolation parameter is not valid
        """
        if not isinstance(dim, tuple):
            raise ValueError(f"The value {dim} for the argument dim is not valid")

        if dim[0] <= 0 or dim[1] <= 0:
            raise ValueError(f"The value {dim} for the argument dim is not valid")

        if interpolation not in ("INTER_LINEAR", "INTER_AREA", "INTER_CUBIC"):
            raise ValueError(f"The value {interpolation} for the argument interpolation is not valid")

        self.__dim = dim

        if interpolation == "INTER_LINEAR":
            self.__interpolation = cv2.INTER_LINEAR
        elif interpolation == "INTER_AREA":
            self.__interpolation = cv2.INTER_AREA
        else:
            self.__interpolation = cv2.INTER_CUBIC

        super().__init__(
            name,
            "resize",
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
            f"Dim: {self.__dim}, Interpolation: {self.__interpolation}",
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
            transformed_images.append(cv2.resize(image, self.__dim, self.__interpolation))

        return transformed_images
