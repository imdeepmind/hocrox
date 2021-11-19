"""Save layer for Hocrox."""
import os
import cv2
import numpy as np


class Save:
    """Save layer for Hocrox."""

    def __init__(self, path, format="npy", name=None):
        """Init method for the Save layer.

        :param name: name of the layer
        :type name: str
        """
        if path and not isinstance(path, str):
            raise ValueError(f"The value {path} for the argument path is not valid")

        if format in ("npy", "img"):
            raise ValueError(f"The value {format} for the argument format is not valid")

        if name and not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        self.__path = path
        self.__format = format
        self.__name = name if name else "Save Layer"

        self.type = "save"
        self.supported_parent_layer = ["resize", "greyscale", "rotate", "crop", "padding", "save"]
        self.bypass_validation = False

    def apply_layer(self, img, name=None):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        filename = f"{self.__name}_{self.type}_{name}"
        if self.__format == "npy":
            np.save(os.path.join(self.__path, filename + ".npy"), img)
        else:
            cv2.imwrite(os.path.join(self.__path, filename), img)

        return img

    def get_description(self):
        """Return layers details for the model to generate summary.

        :return: layer details
        :rtype: str
        """
        return (f"{self.__name}({self.type})", f"Path: {self.__path}, Format: {self.__format}")
