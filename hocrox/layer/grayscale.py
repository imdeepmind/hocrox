"""Resize layer for Hocrox."""
import cv2


class Grayscale:
    """Grayscale layer for Hocrox."""

    def __init__(self, name=None):
        """Init method for the Grayscale layer.

        :param name: name of the layer
        :type name: str
        """
        if name and not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        self.__name = name if name else "Grayscale Layer"

        self.type = "greyscale"
        self.supported_parent_layer = ["resize", "greyscale", "rotate"]
        self.bypass_validation = False

    def apply_layer(self, img):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_description(self):
        """Return layers details for the model to generate summary.

        :return: layer details
        :rtype: str
        """
        return (f"{self.__name}({self.type})", "-")
