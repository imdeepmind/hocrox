"""Convolution layer for Hocrox."""
import numpy as np
import cv2
from hocrox.utils import Layer


class Convolution(Layer):
    """Convolution layer convolves an image with the kernel.

    Here is an example code to use the Convolution layer in a model.

    ```python
    import numpy as np

    from hocrox.model import Model
    from hocrox.layer.preprocessing.transformation import Convolution
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))

    kernel = np.ones((5,5),np.float32)/25
    model.add(Convolution(ddepth=-1, kernel=kernel))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, ddepth, kernel, name=None):
        """Init method for the crop layer.

        Args:
            ddepth (int): Desired depth of the destination image.
            kernel (ndarray): Convolution kernel (or rather a correlation kernel), a single-channel floating point
                matrix; if you want to apply different kernels to different channels, split the image into separate
                color planes using split and process them individually.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the parameter ddepth is not valid
            ValueError: If the parameter kernel is not valid
        """
        if not isinstance(ddepth, int):
            raise ValueError(f"The value {ddepth} for the argument ddepth is not valid")

        if not isinstance(kernel, np.ndarray):
            raise ValueError(f"The value {kernel} for the argument kernel is not valid")

        self.__ddepth = ddepth
        self.__kernel = kernel

        super().__init__(
            name,
            "crop",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Ddepth: {self.__ddepth}, Kernel: {self.__kernel}",
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
            if image is not None and len(image) != 0:
                transformed_image = cv2.filter2D(image, self.__ddepth, self.__kernel)

                if transformed_image is not None and len(transformed_image) != 0:
                    transformed_images.append(transformed_image)

        return transformed_images
