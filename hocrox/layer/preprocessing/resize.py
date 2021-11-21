"""Resize layer for Hocrox."""
import cv2


class Resize:
    """Resize layer for Hocrox."""

    def __init__(self, dim, interpolation="INTER_LINEAR", name=None):
        """Init method for the Resize layer.

        :param dim: dim to resize
        :type dim: tuple
        :param interpolation: interpolation for the resize
        :type interpolation: str
        :param name: name of the layer
        :type name: str
        """
        if not isinstance(dim, tuple):
            raise ValueError(f"The value {dim} for the argument dim is not valid")

        if dim[0] <= 0 and dim[1] <= 0:
            raise ValueError(f"The value {dim} for the argument dim is not valid")

        if interpolation not in ("INTER_LINEAR", "INTER_AREA", "INTER_CUBIC"):
            raise ValueError(f"The value {interpolation} for the argument interpolation is not valid")

        if name and not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        self.__dim = dim
        self.__name = name if name else "Resize Layer"

        if interpolation == "INTER_LINEAR":
            self.__interpolation = cv2.INTER_LINEAR
        elif interpolation == "INTER_AREA":
            self.__interpolation = cv2.INTER_AREA
        else:
            self.__interpolation = cv2.INTER_CUBIC

        self.type = "resize"
        self.supported_parent_layer = [
            "resize",
            "greyscale",
            "rotate",
            "crop",
            "padding",
            "save",
            "horizontal_flip",
            "vertical_flip",
        ]
        self.bypass_validation = False

    def apply_layer(self, images, name=None):
        """Apply the transformation method to change the layer.

        :param img: image for the layer
        :type img: ndarray
        :return: transformed image
        :rtype: ndarray
        """
        transformed_images = []

        for image in images:
            transformed_images.append(cv2.resize(image, self.__dim, self.__interpolation))

        return transformed_images

    def get_description(self):
        """Return layers details for the model to generate summary.

        :return: layer details
        :rtype: str
        """
        return (f"{self.__name}({self.type})", f"Dim: {self.__dim}, Interpolation: {self.__interpolation}")
