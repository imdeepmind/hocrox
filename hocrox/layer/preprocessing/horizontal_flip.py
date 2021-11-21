"""HorizontalFlip layer for Hocrox."""
import cv2


class HorizontalFlip:
    """Rotate layer for Hocrox."""

    def __init__(self, name=None):
        """Init method for the HorizontalFlip layer.

        :param name: name of the layer
        :type name: str
        """
        if name and not isinstance(name, str):
            raise ValueError(f"The value {name} for the argument name is not valid")

        self.__name = name if name else "HorizontalFlip Layer"

        self.type = "horizontal_flip"
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
            transformed_images.append(cv2.flip(image, 1))

        return transformed_images

    def get_description(self):
        """Return layers details for the model to generate summary.

        :return: layer details
        :rtype: str
        """
        return (f"{self.__name}({self.type})", "-")
