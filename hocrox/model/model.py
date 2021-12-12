"""Model class for Hocrox."""

import pickle

from prettytable import PrettyTable
from tqdm import tqdm

from hocrox.utils import is_valid_layer


class Model:
    """Model class is used for making Hocrox models.

    Here is an example code to use the Model class for making a Hocrox model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation import RandomFlip, RandomRotate
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(RandomFlip(number_of_outputs=2))
    model.add(RandomRotate(start_angle=-10.0, end_angle=10.0, number_of_outputs=5))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self):
        """Init method for the Model class."""
        self.__frozen = False
        self.__layers = []

    def add(self, layer):
        """Add a new layer to the model.

        Args:
            layer (layer): Layer class to add into the model.

        Raises:
            ValueError: If the model is frozen.
            ValueError: If the layer is not valid.
            ValueError: If the layer does support the parent layer.
            ValueError: If the first layer is not a read layer.
        """
        if self.__frozen:
            raise ValueError("Model is frozen")

        if not is_valid_layer(layer):
            raise ValueError("The layer is not a valid layer")

        if len(self.__layers) == 0 and layer._get_type() != "read":
            raise ValueError("The first layer needed to be a read layer")

        if len(self.__layers) > 0:
            previous_layer_type = self.__layers[-1]._get_type()

            if not layer._is_valid_child(previous_layer_type):
                tp = layer._get_type()
                raise ValueError(
                    f"The layer of type '{tp}' does not support layer of type '{previous_layer_type}' as parent layer"
                )

        self.__layers.append(layer)

    def summary(self):
        """Generate a summary of the model.

        Here is an example code to use .summary() function in a model.

        ```python
        from hocrox.model import Model

        # Initializing the model
        model = Model()

        ...
        ...

        # Printing the summary of the model
        print(model.summary())
        ```

        Returns:
            str: Summary of the model.
        """
        t = PrettyTable(["Index", "Name", "Parameters"])

        for index, layer in enumerate(self.__layers):
            (name, parameters) = layer._get_description()

            t.add_row([f"#{index+1}", name, parameters])

        return str(t)

    def transform(self):
        """Perform the transformation of the images using the defined model pipeline.

        Here is an example code to use .transform() function in a model.

        ```python
        from hocrox.model import Model

        # Initializing the model
        model = Model()

        ...
        ...

        # Apply transformation to the images based on the defined model pipeline.
        model.transform()
        ```
        """
        read_image_layer = self.__layers[0]
        images, gen = read_image_layer._apply_layer()

        for path, image in tqdm(gen, total=len(images)):
            for layer in self.__layers[1:]:
                image = layer._apply_layer(image, path)

    def freeze(self):
        """Freeze the model. Frozen models cannot be modified.

        Here is an example code to use .freeze() function in a model.

        ```python
        from hocrox.model import Model

        # Initializing the model
        model = Model()

        ...
        ...

        # Freeze the model so it can not be modified
        model.freeze()
        ```
        """
        self.__frozen = True

    def save(self, path):
        """Save the model into the filesystem.

        It internally uses the pickle module to save the model.

        Here is an example code to use .save() function in a model.

        ```python
        from hocrox.model import Model

        # Initializing the model
        model = Model()

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
        model = Model()

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
