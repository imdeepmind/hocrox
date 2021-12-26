"""Image Augmentation is the process of increasing the image data by slightly tweaking and changing the original images.

In other words, Augmentation means artificially creating more data to train ML or DL models.

In image Augmentation, we slightly rotate, flip, change the color of the images to generate multiple versions of the
image. Since all these images are from the original image with slight modifications, they still represent
the original patterns from the original images (most of the time).
"""

from .random_rotate import RandomRotate
from .random_flip import RandomFlip
from .random_zoom import RandomZoom
from .random_brightness import RandomBrightness
from .random_channel_shift import RandomChannelShift
from .random_horizontal_shift import RandomHorizontalShift
from .random_vertical_shift import RandomVerticalShift
