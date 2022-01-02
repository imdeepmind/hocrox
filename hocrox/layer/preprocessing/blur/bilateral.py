"""BilateralBlur layer for Hocrox."""
import cv2

from hocrox.utils import Layer


class BilateralBlur(Layer):
    """BilateralBlur layer blur (image smoothing) an image using an bilateral filter.

    Here is an example code to use the BilateralBlur layer in a model.

    ```python
    from hocrox.model import Model
    from hocrox.layer.preprocessing.blur import BilateralBlur
    from hocrox.layer import Read

    # Initializing the model
    model = Model()

    # Adding model layers
    model.add(Read(path="./img"))
    model.add(BilateralBlur(d=9, sigma_color=75, sigma_space=75))

    # Printing the summary of the model
    print(model.summary())
    ```
    """

    def __init__(self, d, sigma_color, sigma_space, name=None):
        """Init method for BilateralBlur layer.

        Args:
            d (int): Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
                it is computed from sigmaSpace.
            sigma_color (float): Filter sigma in the color space. A larger value of the parameter means that farther
                colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger
                areas of semi-equal color.
            sigma_space (float): Filter sigma in the coordinate space. A larger value of the parameter means that
                farther pixels will influence each other as long as their colors are close enough (see sigmaColor ).
                When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional
                to sigmaSpace.
            name (str, optional): Name of the layer, if not provided then automatically generates a unique name for
                the layer. Defaults to None.

        Raises:
            ValueError: If the d parameter is not valid
            ValueError: If the sigma_color parameter is not valid
            ValueError: If the sigma_space parameter is not valid
        """
        if not isinstance(d, int):
            raise ValueError(f"The value {d} for the argument d is not valid")

        if not (sigma_color, float):
            raise ValueError(f"The value {sigma_color} for the argument sigma_color is not valid")

        if not (sigma_space, float):
            raise ValueError(f"The value {sigma_space} for the argument sigma_space is not valid")

        self.__d = d
        self.__sigma_space = sigma_space
        self.__sigma_color = sigma_color

        super().__init__(
            name,
            "bilateral_blur",
            self.STANDARD_SUPPORTED_LAYERS,
            f"D: {self.__d}, Sigma Space: {self.__sigma_space}, Sigma Color: {self.__sigma_color}",
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
                transformed_image = cv2.bilateralFilter(image, self.__d, self.__sigma_color, self.__sigma_space)

                if transformed_image is not None and len(transformed_image) != 0:
                    transformed_images.append(transformed_image)

        return transformed_images
