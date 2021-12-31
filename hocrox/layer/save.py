"""Save layer for Hocrox."""
import os
import cv2
import numpy as np

from hocrox.utils import Layer


class Save(Layer):
    """Save layer saves images on the local filesystem.

    Here is an example code to use the Save layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer import Read, Save

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(Save(path="./img_to_store", format="npy"))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, path, format="npy", name=None):
        """Init method for the Save layer.

        Args:
            path (str): Path to store the image
            format (str, optional): Format to save the image. Supported formats are npy and img. Defaults to "npy".
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the name parameter is invalid
            ValueError: If the format parameter is invalid
        """
        if path and not isinstance(path, str):
            raise ValueError(f"The value {path} for the argument path is not valid")

        if format not in ("npy", "img"):
            raise ValueError(f"The value {format} for the argument format is not valid")

        self.__path = path
        self.__format = format

        super().__init__(
            name,
            "save",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Path: {self.__path}, Format: {self.__format}",
        )

    def _apply_layer(self, images, name=None):
        """Apply the transformation method to change the layer.

        Args:
            images (list[ndarray]): List of images to transform.
            name (str, optional): Name of the image series, used for saving the images. Defaults to None.

        Returns:
            list[ndarray]: Return the transform images
        """
        for index, image in enumerate(images):
            if image is not None and len(image) != 0:
                layer_name = self._get_name()
                filename = f"{layer_name}_{index}_{name}"

                if self.__format == "npy":
                    np.save(os.path.join(self.__path, filename + ".npy"), image)
                else:
                    cv2.imwrite(os.path.join(self.__path, filename), image)

        return images
