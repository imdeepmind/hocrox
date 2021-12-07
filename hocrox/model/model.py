"""Model class for Hocrox."""

import pickle
import cv2
import os

from prettytable import PrettyTable
from tqdm import tqdm

from hocrox.utils import is_valid_layer


class Model:
    """Model class is used for making Hocrox models.

    Models are the heart of Hocrox. In Hocrox, the model contains the entire pipeline for preprocessing and/or
    augmenting the image.

    Here is an example code to use the Model class for making a Hocrox model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation import RandomFlip, RandomRotate

    # Initializing the model
    model = Model("./img")

    # Adding model layers
    model.add(RandomFlip(number_of_outputs=2))
    model.add(RandomRotate(start_angle=-10.0, end_angle=10.0, number_of_outputs=5))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, read_dir):
        """Init method for the Model class.

        Args:
            read_dir (str): Path for read the image folder. Please note that the path need to contain only valid images
                and no folders or other files.

        Raises:
            ValueError: If the path is not valid.
        """
        if not isinstance(read_dir, str):
            raise ValueError("Please provide a valid read_dir path")

        self.__frozen = False
        self.__layers = []
        self.__read_dir = read_dir

    def __read_image_gen(self, images):
        """Read images from the filesystem and returns a generator.

        Args:
            images (list): List of images to read

        Yields:
            ndarray: Image in the form of numpy ndarray.
        """
        for path in images:
            image = cv2.imread(os.path.join(self.__read_dir, path), 1)

            yield path, [image]

    def add(self, layer):
        """Add a new layer to the model.

        Args:
            layer (layer): Layer class to add into the model.

        Raises:
            ValueError: If the model is frozen.
            ValueError: If the layer is not valid.
            ValueError: If the layer does support the parent layer.
        """
        if self.__frozen:
            raise ValueError("Model is frozen")

        if not is_valid_layer(layer):
            raise ValueError("The layer is not a valid layer")

        if len(self.__layers) > 0:
            previous_layer_type = self.__layers[-1]._get_type()

            if not layer._is_valid_child(previous_layer_type):
                tp = layer._get_type()
                raise ValueError(
                    f"The layer of type '{tp}' does not support layer of type '{previous_layer_type}' as parent layer"
                )

        self.__layers.append(layer)

    def summary(self):
        """Generate summary of the model.

        Here is an example code to use .summary() function in a model.

        ```python
        from hocrox.model import Model

        # Initializing the model
        model = Model("./img")

        ...
        ...

        # Printing the summary of the model
        print(model.summary())
        ```

        Returns:
            sre: Summary of the model.
        """
        t = PrettyTable(["Index", "Name", "Parameters"])

        for index, layer in enumerate(self.__layers):
            (name, parameters) = layer._get_description()

            t.add_row([f"#{index+1}", name, parameters])

        return str(t)

    def transform(self):
        """Perform the transformation of the images using the defined the model pipeline.

        Here is an example code to use .transform() function in a model.

        ```python
        from hocrox.model import Model

        # Initializing the model
        model = Model("./img")

        ...
        ...

        # Apply transformation to the images based on the defined model pipeline.
        model.transform()
        ```
        """
        images = os.listdir(self.__read_dir)
        gen = self.__read_image_gen(images)

        for path, image in tqdm(gen, total=len(images)):
            for layer in self.__layers:
                image = layer._apply_layer(image, path)

    def freeze(self):
        """Freeze the model. Frozen models cannot be modified.

        Here is an example code to use .freeze() function in a model.

        ```python
        from hocrox.model import Model

        # Initializing the model
        model = Model("./img")

        ...
        ...

        # Freeze the model so it can not be modified
        model.freeze()
        ```
        """
        self.__frozen = True

    def save(self, path):
        """Save the defined pipeline for filesystem.

        It internally uses pickle to save the model.

        Here is an example code to use .save() function in a model.

        ```python
        from hocrox.model import Model

        # Initializing the model
        model = Model("./img")

        ...
        ...

        # Save the model to specific path
        model.save(path="./model.hocrox")
        ```

        Args:
            path (str): Path to store the model.

        Raises:
            ValueError: If the path is not valid.
        """
        if not isinstance(path, str):
            raise ValueError("Path is not valid")

        model_config = {"frozen": self.__frozen, "layers": self.__layers}

        with open(path, "wb") as f:
            pickle.dump(model_config, f)

    def load(self, path):
        """Load a model from the filesystem.

        Here is an example code to use .load() function in a model.

        ```python
        from hocrox.model import Model

        # Initializing the model
        model = Model("./img")

        ...
        ...

        # Load the model to specific path
        model.load(path="./model.hocrox")
        ```

        Args:
            path (str): Path to read the model from.

        Raises:
            ValueError: If the path is not valid.
        """
        if not isinstance(path, str):
            raise ValueError("Path is not valid")

        with open(path, "rb") as f:
            model_config = pickle.load(f)

            self.__layers = model_config["layers"]
            self.__frozen = model_config["frozen"]
