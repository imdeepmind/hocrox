"""VerticalShift layer for Hocrox."""
import cv2

from hocrox.utils import Layer

__all__ = ["VerticalShift"]


class VerticalShift(Layer):
    """VerticalShift layer shifts the image vertically.

    Here is an example code to use the VerticalShift layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation.shift import VerticalShift
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(VerticalShift(by=0.7))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, by=0.7, name=None):
        """Init method for the VerticalShift layer.

        Args:
            by (float, optional): Rate of the shift, 0.0 means no change. Defaults to 0.7.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the by parameter is not valid
        """
        if not (isinstance(by, float)):
            raise ValueError(f"The value {by} for the argument by is not valid")

        super().__init__(
            name,
            "vertical_shift",
            self.STANDARD_SUPPORTED_LAYERS,
            f"By:{by}",
        )

        self.__by = by

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
                transformed_image = self.__vertical_shift(image, self.__by)

                if transformed_image is not None and len(transformed_image) != 0:
                    transformed_images.append(transformed_image)

        return transformed_images

    @staticmethod
    def __vertical_shift(img, by):
        """Apply horizontal_shift function to the image.

        Args:
            img (ndarray): Image to change the change
            by (float): Rate of change

        Returns:
            ndarray: Updated image
        """
        h, w = img.shape[:2]
        to_shift = h * by

        if by > 0:
            img = img[: int(h - to_shift), :, :]
        if by < 0:
            img = img[int(-1 * to_shift) :, :, :]

        if img is None or len(img) == 0:
            return img

        img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)

        return img
