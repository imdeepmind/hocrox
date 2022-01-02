"""RandomVerticalShift layer for Hocrox."""
import random
import cv2

from hocrox.utils import Layer


class RandomVerticalShift(Layer):
    """RandomVerticalShift layer randomly shifts the image vertically.

    Here is an example code to use the RandomVerticalShift layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation.shift import RandomVerticalShift
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(RandomVerticalShift(ratio=0.7, number_of_outputs=1))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, ratio=0.7, probability=1.0, number_of_outputs=1, name=None):
        """Init method for the RandomVerticalShift layer.

        Args:
            ratio (float, optional): Ratio is used to define the range of the shift. Defaults to 0.7.
            probability (float, optional): Probability rate for the layer, if the rate of 0.5 then the layer is applied
                on 50% of the images. Defaults to 1.0.
            number_of_outputs (int, optional): Number of images to output. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the ratio parameter is not valid
            ValueError: If the probability parameter is not valid
            ValueError: If the number_of_images parameter is not valid
        """
        if not (isinstance(ratio, float)):
            raise ValueError(f"The value {ratio} for the argument ratio is not valid")

        if not isinstance(probability, float) or probability < 0.0 or probability > 1.0:
            raise ValueError(f"The value {probability} for the argument probability is not valid")

        if not isinstance(number_of_outputs, int) or number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        super().__init__(
            name,
            "random_vertical_shift",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Ratio:{ratio}, Probability: {probability}, Number of Outputs: {number_of_outputs}",
        )

        self.__number_of_outputs = number_of_outputs
        self.__ratio = ratio
        self.__probability = probability

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
            for _ in range(self.__number_of_outputs):
                should_perform = self._get_probability(self.__probability)

                if image is not None and len(image) != 0:
                    transformed_image = self.__vertical_shift(image, self.__ratio) if should_perform else image

                    if transformed_image is not None and len(transformed_image) != 0:
                        transformed_images.append(transformed_image)

        return transformed_images

    @staticmethod
    def __vertical_shift(img, ratio):
        """Apply horizontal_shift function to the image.

        Args:
            img (ndarray): Image to change the brightness
            ratio (float): High range of the brightness

        Returns:
            ndarray: Updated image
        """
        ratio = random.uniform(-ratio, ratio)

        h, w = img.shape[:2]
        to_shift = h * ratio

        if ratio > 0:
            img = img[: int(h - to_shift), :, :]
        if ratio < 0:
            img = img[int(-1 * to_shift) :, :, :]

        if img is None or len(img) == 0:
            return img

        img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)

        return img
