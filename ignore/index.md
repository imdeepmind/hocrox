# Introduction

Hocrox is an image preprocessing and augmentation library. It provides a [Keras](https://keras.io/) like simple interface to make preprocessing and augmentation pipelines. Hocrox internally uses [OpenCV](https://opencv.org/) to perform the operations on images. OpenCV is one of the most popular for Computer Vision library.

Here are some of the highlights of Hocrox:

- Provides an easy interface that is suitable for radio pipeline development
- It internally uses OpenCV
- Highly configurable with support for custom layers

# The Keas interface

Keras is one of the most popular Deep Learning library. Keras provides a very simple yet powerful interface that can be used to develop start-of-the-art Deep Learning models.

Check the code below. This is a simple Keras code to make a simple neural network.

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

In Hocrox, the interface for making pipelines very much similar. So anyone can make complex pipelines with few lines of code.

# Documentation

Documentation for Hocrox is available [here](https://imdeepmind.com/hocrox/documentation).

# Example

Here is one simple pipeline for preprocessing images.

```python
from hocrox.model import Model
from hocrox.layer.preprocessing import Resize, Grayscale, Padding, Save

# Initializing the model
model = Model("./img")

# Adding model layers
model.add(Resize((100, 100), name="Resize Layer"))
model.add(Grayscale(name="Grayscale Layer"))
model.add(Padding(10, 20, 70, 40, [255, 255, 255], name="Padding Layer"))
model.add(Save("img2/", name="Save image"))

# Printing the summary of the model
print(model.summary())

# Apply transform to the images
model.transform()
```

# Contributors

Check the list of contributors [here](https://github.com/imdeepmind/hocrox/graphs/contributors).

# License

[MIT](https://github.com/imdeepmind/hocrox/blob/main/LICENSE)