"""Brightness layer for Hocrox."""

from hocrox.layer.augmentation.color import RandomBrightness


class Brightness(RandomBrightness):
    """Brightness layer changes the brightness of the image based on the provided low and high.

    Here is an example code to use the Brightness layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation import Brightness
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(Brightness(low=0.5, high=3.0))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, low=0.5, high=3.0, name=None):
        """Init method for the Brightness layer.

        Args:
            low (float, optional): Starting range of the brightness. Defaults to 0.5.
            high (float, optional): Ending range of the brightness. Defaults to 3.0.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the low parameter is not valid
            ValueError: If the high parameter is not valid
        """
        super().__init__(low=low, high=high, probability=1.0, number_of_outputs=1, name=name)
