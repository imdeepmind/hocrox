"""Resize layer for Hocrox."""
import cv2


class Resize:
    """Resize layer for Hocrox."""

    def __init__(self, dim, interpolation, name):
        """Init method for the Resize layer.

        :param dim: dim to resize
        :type dim: tuple
        :param interpolation: interpolation for the resize
        :type interpolation: str
        :param name: name of the layer
        :type name: str
        """
        self.__dim = dim
        self.__interpolation = interpolation
        self.__name = name

        self.type = "resize"
        self.supported_parent_layer = ["resize"]
        self.bypass_validation = False

    def apply_layer(self, img):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        return cv2.resize(img, self.__dim, self.__interpolation)

    def get_description(self):
        """Return layers details for the model to generate summary.

        :return: layer details
        :rtype: str
        """
        return (f"{self.__name}({self.type})", f"Dim: {self.__dim}, Interpolation: {self.__interpolation}")
