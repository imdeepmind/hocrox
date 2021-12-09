"""Layers are the fundamental blocks of Hocrox image preprocessing and augmentation library.

In Hocrox, each layer basically means a function that will be performed on the images to preprocess or/and augment.

Currently, in Hocrox, there are several layers. The complete list is below.

- Preprocessing
    - Crop
    - Grayscale
    - Horizontal Flip
    - Padding
    - Resize
    - Rotate
    - Save
    - Vertical Flip
- Augmentation
    - RandomFlip
    - RandomRotate
"""

from . import preprocessing
from . import augmentation
