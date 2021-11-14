import cv2


class Resize:
    def __init__(self, dim, interpolation, name):
        self.__dim = dim
        self.__interpolation = interpolation
        self.__name = name
        self.__type = "resize"

    def apply_layer(self, img):
        return cv2.resize(img, self.__dim, self.__interpolation)

    def get_description(self):
        return (
            f"{self.__name}({self.__type})",
            f"Dim: {self.__dim}, Interpolation: {self.__interpolation}",
        )
