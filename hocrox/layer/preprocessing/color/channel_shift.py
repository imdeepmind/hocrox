"""ChannelShift layer for Hocrox."""

from hocrox.layer.augmentation.color import RandomChannelShift


class ChannelShift(RandomChannelShift):
    """ChannelShift layer randomly adds some value to the channels in the image.

    Here is an example code to use the ChannelShift layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation import ChannelShift
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(ChannelShift(low=1, high=5))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, low=1, high=5, name=None):
        """Init method for the ChannelShift layer.

        Args:
            low (int, optional): Starting range of the brightness. Defaults to 0.5.
            end (int, optional): Ending range of the brightness. Defaults to 3.0.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the low parameter is not valid
            ValueError: If the high parameter is not valid
        """
        super().__init__(low=low, high=high, probability=1.0, number_of_outputs=1, name=name)
