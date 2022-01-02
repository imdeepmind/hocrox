"""RandomRotate layer for Hocrox."""
import cv2
import random
import numpy as np

from hocrox.utils import Layer


class RandomRotate(Layer):
    """RandomRotate layer randomly rotates an image to a certain angle.

    Here is an example code to use the RandomRotate layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation.transformation import RandomRotate
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(RandomRotate(start_angle=-10.0, end_angle=10.0, number_of_outputs=5))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    @staticmethod
    def __rotate_image(image, angle):
        """Rotate an image to certain angle.

        Args:
            image (ndarray): Image to rotate
            angle (float): Angle to rotate the image

        Returns:
            ndarray: New rotated image
        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return result

    def __init__(self, start_angle, end_angle, probability=1.0, number_of_outputs=1, name=None):
        """Init method for the RandomRotate layer.

        Args:
            start_angle (float): Start of the range of angle
            end_angle (float): End of the range of angle
            probability (float, optional): Probability rate for the layer, if the rate of 0.5 then the layer is applied
                on 50% of the images. Defaults to 1.0.
            number_of_outputs (int, optional): Number of images to output. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the start_angle parameter is not valid
            ValueError: If the end_angle parameter is not valid
            ValueError: If the probability parameter is not valid
            ValueError: If the number_of_outputs parameter is not valid
        """
        if not (isinstance(start_angle, int) or isinstance(start_angle, float)):
            raise ValueError(f"The value {start_angle} for the argument start_angle is not valid")

        if not (isinstance(end_angle, int) or isinstance(end_angle, float)):
            raise ValueError(f"The value {end_angle} for the argument end_angle is not valid")

        if start_angle > end_angle:
            raise ValueError(f"The value {start_angle} for the argument start_angle is not valid")

        if not isinstance(probability, float) or probability < 0.0 or probability > 1.0:
            raise ValueError(f"The value {probability} for the argument probability is not valid")

        if not isinstance(number_of_outputs, int) or number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        self.__start_angle = start_angle
        self.__end_angle = end_angle
        self.__probability = probability
        self.__number_of_outputs = number_of_outputs

        super().__init__(
            name,
            "random_rotate",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Probability: {probability}, Number of Outputs: {number_of_outputs}",
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
            for _ in range(self.__number_of_outputs):
                angle = random.uniform(self.__start_angle, self.__end_angle)
                should_perform = self._get_probability(self.__probability)

                if image is not None and len(image) != 0:
                    transformed_image = self.__rotate_image(image, angle) if should_perform else image

                    if transformed_image is not None and len(transformed_image) != 0:
                        transformed_images.append(transformed_image)

        return transformed_images
