"""RandomZoom layer for Hocrox."""
import cv2
import random

from hocrox.utils import Layer


class RandomZoom(Layer):
    """RandomZoom layer randomly zooms an image based on the defined zoom range.

    Here is an example code to use the RandomZoom layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.augmentation.transformation import RandomZoom
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(RandomZoom(start=0, end=1, number_of_outputs=1))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, start=0.0, end=1.0, probability=1.0, number_of_outputs=1, name=None):
        """Init method for the RandomZoom layer.

        Args:
            start (float, optional): Starting range of the zoom, the value should be between 0 and 1. Defaults to 0.
            end (float, optional): Ending range of the zoom, the value should be between 0 and 1. Defaults to 1.
            probability (float, optional): Probability rate for the layer, if the rate of 0.5 then the layer is applied
                on 50% of the images. Defaults to 1.0.
            number_of_outputs (int, optional): Number of images to output. Defaults to 1.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the start parameter is not valid
            ValueError: If the end parameter is not valid
            ValueError: If the number_of_images parameter is not valid
            ValueError: If the probability parameter is not valid
        """
        if not (isinstance(start, float) and start >= 0 and start < 1):
            raise ValueError(f"The value {start} for the argument start is not valid")

        if not (isinstance(end, float) and end > 0 and end <= 1):
            raise ValueError(f"The value {end} for the argument end is not valid")

        if not isinstance(probability, float) or probability < 0.0 or probability > 1.0:
            raise ValueError(f"The value {probability} for the argument probability is not valid")

        if not isinstance(number_of_outputs, int) or number_of_outputs < 1:
            raise ValueError(f"The value {number_of_outputs} for the argument number_of_outputs is not valid")

        super().__init__(
            name,
            "random_zoom",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Start: {start}, end:{end}, Probability: {probability}, Number of Outputs: {number_of_outputs}",
        )

        self.__number_of_outputs = number_of_outputs
        self.__start = start
        self.__end = end
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
                    transformed_image = self.__zoom(image, self.__start, self.__end) if should_perform else image

                    if transformed_image is not None and len(transformed_image) != 0:
                        transformed_images.append(transformed_image)

        return transformed_images

    @staticmethod
    def __zoom(img, start, end):
        """Zoom the image.

        Args:
            img (ndarray): Image to zoom
            start (float): Start range of the zoom
            end (float): End range of the zoom

        Returns:
            ndarray: Zoomed image
        """
        zoom_value = random.uniform(start, end)

        h, w = img.shape[:2]

        h_taken = int(zoom_value * h)
        w_taken = int(zoom_value * w)

        h_start = random.randint(0, h - h_taken)
        w_start = random.randint(0, w - w_taken)

        img = img[h_start : h_start + h_taken, w_start : w_start + w_taken, :]

        if img is None or len(img) == 0:
            return img

        img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)

        return img
