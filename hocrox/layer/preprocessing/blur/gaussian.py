"""GaussianBlur layer for Hocrox."""
import cv2

from hocrox.utils import Layer


class GaussianBlur(Layer):
    """GaussianBlur layer blur (image smoothing) an image using a gaussian filter.

    Here is an example code to use the GaussianBlur layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing.blur import GaussianBlur
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(GaussianBlur(kernel_size=(5,5), sigma_x=0, sigma_y=0)))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, kernel_size, sigma_x, sigma_y=0, name=None):
        """Init method for GaussianBlur layer.

        Args:
            kernel_size (tuple): Kernel size for the filter
            sigma_x (float): Gaussian kernel standard deviation in X direction.
            sigma_y (float, optional): Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set
                to be equal to sigmaX, if both sigmas are zeros, they are computed from
                kernel_size.width and kernel_size.height, respectively.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the kernel_size parameter is not valid
            ValueError: If the sigma_x parameter is not valid
            ValueError: If the sigma_y parameter is not valid
        """
        if not isinstance(kernel_size, tuple):
            raise ValueError(f"The value {kernel_size} for the argument kernel_size is not valid")

        if not (sigma_x, float):
            raise ValueError(f"The value {sigma_x} for the argument sigma_x is not valid")

        if not (sigma_y, float):
            raise ValueError(f"The value {sigma_y} for the argument sigma_y is not valid")

        self.__kernel_size = kernel_size
        self.__sigma_x = sigma_x
        self.__sigma_y = sigma_y

        super().__init__(
            name,
            "gaussian_blur",
            self.STANDARD_SUPPORTED_LAYERS,
            f"Kernel Size: {self.__kernel_size}, Sigma X: {self.__sigma_x}, Sigma Y: {self.__sigma_y}",
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
                transformed_image = cv2.GaussianBlur(image, self.__kernel_size, self.__sigma_x, self.__sigma_y)

                if transformed_image is not None and len(transformed_image) != 0:
                    transformed_images.append(transformed_image)

        return transformed_images
