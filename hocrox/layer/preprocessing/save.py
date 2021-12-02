"""Save layer for Hocrox."""
import os
import cv2
import numpy as np

from hocrox.utils import Layer


class Save(Layer):
    """Save layer for Hocrox."""

    def __init__(self, path, format="npy", name=None):
        """Init method for the Save layer.

        :param name: name of the layer
        :type name: str
        """
        if path and not isinstance(path, str):
            raise ValueError(f"The value {path} for the argument path is not valid")

        if format not in ("npy", "img"):
            raise ValueError(f"The value {format} for the argument format is not valid")

        if name and not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        self.__path = path
        self.__format = format

        super().__init__(
            name,
            "save",
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
            f"Path: {self.__path}, Format: {self.__format}",
        )

    def _apply_layer(self, images, name=None):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        for index, image in enumerate(images):
            layer_name = self.get_name()
            filename = f"{layer_name}_{index}_{name}"

            if self.__format == "npy":
                np.save(os.path.join(self.__path, filename + ".npy"), image)
            else:
                cv2.imwrite(os.path.join(self.__path, filename), image)

        return images
