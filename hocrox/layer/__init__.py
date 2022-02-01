"""Layers are the fundamental blocks of Hocrox.

In Hocrox, each layer basically a function that will be performed on the images to preprocess or augment the image.

Currently, in Hocrox, there are several supported layers. The complete list is below.

- Save
- Read
- Preprocessing
    - Blur
        - AverageBlur
        - GaussianBlur
        - MedianBlur
        - BilateralBlur
    - Color
        - Brightness
        - ChannelShift
        - Grayscale
        - Rescale
    - Flip
        - Horizontal Flip
        - Vertical Flip
    - Shift
        - Horizontal Shift
        - Vertical Shift
    - Transformation
        - Crop
        - Padding
        - Resize
        - Rotate
        - Convolution
- Augmentation
    - Color
        - RandomBrightness
        - RandomChannelShift
    - Flip
        - RandomVerticalFlip
        - RandomHorizontalFlip
        - RandomFlip
    - Shift
        - RandomHorizontalShift
        - RandomVerticalShift
    - Transformation
        - RandomRotate
        - RandomZoom
"""

from . import preprocessing
from . import augmentation
from .read import Read
from .save import Save
