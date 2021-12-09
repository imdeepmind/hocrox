"""Image Augmentation is the process of increasing the image data by slightly tweaking and changing the original images.

In other words, Augmentation means artificially creating more data to train ML or DL models.

In image Augmentation, we slightly rotate, flip, change the color of the images to generate multiple versions of the
image. Since all these images are from the original image with slight modifications, they still represent
the original patterns from the original images (most of the time).

In Hocrox, currently, there are only 2 augmentation layers, RandomFlip and RandomRotate.
"""

from .random_rotate import RandomRotate
from .random_flip import RandomFlip
