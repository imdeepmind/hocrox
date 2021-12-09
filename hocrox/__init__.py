"""Hocrox is an image preprocessing and augmentation library.

It provides a [Keras](https://keras.io/) like simple interface to make preprocessing and augmentation pipelines.
Hocrox internally uses [OpenCV](https://opencv.org/) to perform the operations on images. OpenCV is one of the most \
popular Computer Vision library.

Here are some of the highlights of Hocrox:

- Provides an easy interface that is suitable for radio pipeline development
- It internally uses OpenCV
- Highly configurable with support for custom layers
"""

from . import model
from . import layer
from . import utils
