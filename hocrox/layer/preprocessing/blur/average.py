"""AverageBlur layer for Hocrox."""
import cv2

from hocrox.utils import Layer


class AverageBlur(Layer):
    """AverageBlur layer blur (image smoothing) an image using a normalized box filter.

    Here is an example code to use the AverageBlur layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing.blur import AverageBlur
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(AverageBlur(kernel_size=(5,5)))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, kernel_size, name=None):
        """Init method for AverageBlur layer.

        Args:
            kernel_size (tuple): Kernel size for the filter
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the kernel_size parameter is not valid
        """
        if not isinstance(kernel_size, tuple):
            raise ValueError(f"The value {kernel_size} for the argument kernel_size is not valid")

        self.__kernel_size = kernel_size

        super().__init__(
            name,
            "average_blur",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Kernel Size: {self.__kernel_size}",
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
                transformed_image = cv2.blur(image, self.__kernel_size)

                if transformed_image is not None and len(transformed_image) != 0:
                    transformed_images.append(transformed_image)

        return transformed_images
