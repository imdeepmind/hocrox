"""HorizontalShift layer for Hocrox."""

from hocrox.layer.augmentation.shift import RandomHorizontalShift


class HorizontalShift(RandomHorizontalShift):
    """HorizontalShift layer randomly shifts the image horizontally.

    Here is an example code to use the HorizontalShift layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation import HorizontalShift
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(HorizontalShift(ratio=0.7))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, ratio=0.7, name=None):
        """Init method for the HorizontalShift layer.

        Args:
            ratio (float, optional): Ratio is used to define the range of the shift. Defaults to 0.7.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the ratio parameter is not valid
        """
        super().__init__(ratio=ratio, probability=1.0, number_of_outputs=1, name=name)
