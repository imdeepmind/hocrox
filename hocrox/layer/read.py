"""Read layer for Hocrox."""
import os
import cv2

from hocrox.utils import Layer


class Read(Layer):
    """Read layer reads images from the local filesystem.

    Here is an example code to use the Read layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, path, name=None):
        """Init method for the Read layer.

        Args:
            path (str): Path to store the image
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the name parameter is invalid
        """
        if path and not isinstance(path, str):
            raise ValueError(f"The value {path} for the argument path is not valid")

        self.__path = path

        super().__init__(
            name,
            "read",
            [],  # Read layer does not support any parent layers
            f"Path: {self.__path}",
        )

    def __read_image_gen(self, images):
        """Read images from the filesystem and returns a generator.

        Args:
            images (list): List of images to read

        Yields:
            ndarray: Image in the form of numpy ndarray.
        """
        for path in images:
            image = cv2.imread(os.path.join(self.__path, path), 1)

            yield path, [image]

    def _apply_layer(self):
        """Apply the transformation method to change the layer.

        Returns:
            tuple: List of images and a generator function to read the image once at a time.
        """
        images = os.listdir(self.__path)
        gen = self.__read_image_gen(images)

        return images, gen
