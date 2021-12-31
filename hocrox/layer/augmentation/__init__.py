"""Image Augmentation is the process of modifying existing images and creating multiple copies of it, hence increasing \
    total number of images.

In image Augmentation, we slightly rotate, flip, change the color of the images to generate multiple versions of the
image. Since all these images are from the original image with slight modifications, they still represent
the original patterns from the original images (most of the time).
"""

from . import color
from . import transformation
from . import shift
from . import flip
