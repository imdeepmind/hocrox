import pickle
import inspect
import numpy as np
import cv2
import os

from prettytable import PrettyTable


class Sequential:
    def __init__(self, read_dir, output_dir):
        if not isinstance(read_dir, str):
            raise ValueError("Please provide a valid read_dir path")

        if not isinstance(output_dir, str):
            raise ValueError("Please provide a valid output_dir path")

        self.__frozen = False
        self.__layers = []
        self.__read_dir = read_dir
        self.__output_dir = output_dir

    def read_image_gen(self, images):
        for image in images:
            img = cv2.imread(os.path.join(self.__read_dir, image), 1)

            yield image, img

    def save_image(self, path, image):
        cv2.imwrite(os.path.join(self.__output_dir, path), image)

    def add(self, layer):
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

    def transform(self):
        images = os.listdir(self.__read_dir)
        gen = self.read_image_gen(images)

        for path, image in gen:
            for layer in self.__layers:
                image = layer.apply_layer(image)

            self.save_image(path, image)

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
