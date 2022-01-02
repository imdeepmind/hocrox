"""RandomFlip layer for Hocrox."""
import cv2
import random

from hocrox.utils import Layer


class RandomFlip(Layer):
    """RandomFlip layer randomly flips an image vertically or horizontally.

    Here is an example code to use the RandomFlip layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation.flip import RandomFlip
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(RandomFlip(number_of_outputs=1))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, probability=1.0, number_of_outputs=1, name=None):
        """Init method for the RandomFlip layer.

        Args:
            probability (float, optional): Probability rate for the layer, if the rate of 0.5 then the layer is applied
                on 50% of the images. Defaults to 1.0.
            number_of_outputs (int, optional): Number of images to output. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the probability parameter is not valid
            ValueError: If the number_of_images parameter is not valid
        """
        if not isinstance(probability, float) or probability < 0.0 or probability > 1.0:
            raise ValueError(f"The value {probability} for the argument probability is not valid")

        if not isinstance(number_of_outputs, int) or number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        super().__init__(
            name,
            "random_flip",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Probability: {probability}, Number of Outputs: {number_of_outputs}",
        )

        self.__number_of_outputs = number_of_outputs
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
                flip = random.randint(0, 1)
                should_perform = self._get_probability(self.__probability)

                if image is not None and len(image) != 0:
                    transformed_image = cv2.flip(image, flip) if should_perform else image

                    if transformed_image is not None and len(transformed_image) != 0:
                        transformed_images.append(transformed_image)

        return transformed_images
