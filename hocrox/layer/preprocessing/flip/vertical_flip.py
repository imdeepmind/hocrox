"""VerticalFlip layer for Hocrox."""
import cv2

from hocrox.utils import Layer


class VerticalFlip(Layer):
    """VerticalFlip layer vertically flips an image.

    Here is an example code to use the Crop layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing.flip import VerticalFlip
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(VerticalFlip())

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, name=None):
        """Init method for horizontal flip layer.

        Args:
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        """
        super().__init__(
            name,
            "vertical_flip",
            self.STANDARD_SUPPORTED_LAYERS,
            "-",
        )

    def _apply_layer(self, images, name=None):
        """Apply the transformation method to change the layer.

        Args:
            images (list[ndarray]): List of images to transform.
            name (str, optional): Name of the image series, used for saving the images. Defaults to None.

        Returns:
            list[ndarray]: Return the transform images
        """
        transformed_images = []

        for image in images:
            if image is not None and len(image) != 0:
                transformed_image = cv2.flip(image, 0)

                if transformed_image is not None and len(transformed_image) != 0:
                    transformed_images.append(transformed_image)

        return transformed_images
