import pickle
import numpy as np

from prettytable import PrettyTable


class Sequential:
    def __init__(self):
        self.__frozen = False
        self.__layers = []

    def add(self, layer):
        if self.__frozen:
            raise ValueError("Model is frozen")

        if not (hasattr(layer, "apply_layer") and hasattr(layer, "get_description")):
            raise ValueError("The layer is not a valid layer")

        self.__layers.append(layer)

    def summary(self):
        t = PrettyTable(["Index", "Name", "Parameters"])

        for index, layer in enumerate(self.__layers):
            (
                name,
                parameters,
            ) = layer.get_description()

            t.add_row(
                [
                    f"#{index+1}",
                    name,
                    parameters,
                ]
            )

        return str(t)

    def transform(self, images):
        if not isinstance(images, list):
            raise ValueError(
                "Invalid images, images needed to be a list of numpy array"
            )

        transformed_images = []

        for image in images:
            if not isinstance(image, np.ndarray):
                raise ValueError("Invalid image, image needed to an numpy array")

            for layer in self.__layers:
                image = layer.apply_layer(image)

            transformed_images.append(image)

        return transformed_images

    def transform_generator(self):
        pass

    def freeze(self):
        self.__frozen = True

    def save(self, path):
        if not isinstance(path, str):
            raise ValueError("Path is not valid")

        model_config = {"frozen": self.__frozen, "layers": self.__layers}

        with open(path, "wb") as f:
            pickle.dump(model_config, f)

    def load(self, path):
        if not isinstance(path, str):
            raise ValueError("Path is not valid")

        with open(path, "rb") as f:
            model_config = pickle.load(f)

            self.__layers = model_config["layers"]
            self.__frozen = model_config["frozen"]
