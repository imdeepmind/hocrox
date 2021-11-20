"""Model model for Hocrox."""

import pickle
import cv2
import os

from prettytable import PrettyTable


class Model:
    """Model model for Hocrox."""

    def __init__(self, read_dir):
        """Init method for the Model model.

        :param read_dir: path where the images are stored
        :type read_dir: str
        :raises ValueError: when the read_dir is not valid
        """
        if not isinstance(read_dir, str):
            raise ValueError("Please provide a valid read_dir path")

        self.__frozen = False
        self.__layers = []
        self.__read_dir = read_dir

    def __read_image_gen(self, images):
        """Create a generator function for the image.

        :param images: list of all images
        :type images: list of str
        :yield: one image array
        :rtype: ndarray
        """
        for image in images:
            img = cv2.imread(os.path.join(self.__read_dir, image), 1)

            yield image, img

    def add(self, layer):
        """Add new layers to the model.

        :param layer: layer to add in the model
        :type layer: class
        :raises ValueError: if the model is frozen
        :raises ValueError: if the layer is not valid
        :raises ValueError: if the layer is not valid
        """
        if self.__frozen:
            raise ValueError("Model is frozen")

        if not (hasattr(layer, "apply_layer") and hasattr(layer, "get_description")):
            raise ValueError("The layer is not a valid layer")

        if len(self.__layers) > 0 and not layer.bypass_validation:
            previous_layer_type = self.__layers[-1].type
            if previous_layer_type not in layer.supported_parent_layer:
                tp = layer.type
                raise ValueError(
                    f"The layer of type '{tp}' does not support layer of type '{previous_layer_type}' as parent layer"
                )

        self.__layers.append(layer)

    def summary(self):
        """Generate the summary for the model.

        :return: summary of the model
        :rtype: str
        """
        t = PrettyTable(["Index", "Name", "Parameters"])

        for index, layer in enumerate(self.__layers):
            (name, parameters) = layer.get_description()

            t.add_row([f"#{index+1}", name, parameters])

        return str(t)

    def transform(self):
        """Transform the images using the layers."""
        images = os.listdir(self.__read_dir)
        gen = self.__read_image_gen(images)

        for path, image in gen:
            for layer in self.__layers:
                image = layer.apply_layer(image, path)

    def freeze(self):
        """Freeze the model so it cannot be edited."""
        self.__frozen = True

    def save(self, path):
        """Save the model to filesystem.

        :param path: path where the model will be stored
        :type path: str
        :raises ValueError: if the model is not valid
        """
        if not isinstance(path, str):
            raise ValueError("Path is not valid")

        model_config = {"frozen": self.__frozen, "layers": self.__layers}

        with open(path, "wb") as f:
            pickle.dump(model_config, f)

    def load(self, path):
        """Load model from fiesystem.

        :param path: path where the model is stored
        :type path: str
        :raises ValueError: if the path is not valid
        """
        if not isinstance(path, str):
            raise ValueError("Path is not valid")

        with open(path, "rb") as f:
            model_config = pickle.load(f)

            self.__layers = model_config["layers"]
            self.__frozen = model_config["frozen"]
