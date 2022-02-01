# Hocrox

An image preprocessing and augmentation library with Keras like interface.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Rwh0VMr6A3E/0.jpg)](https://www.youtube.com/watch?v=Rwh0VMr6A3E)

[![Hocrox Code Check](https://github.com/imdeepmind/hocrox/actions/workflows/build_check.yml/badge.svg)](https://github.com/imdeepmind/hocrox/actions/workflows/build_check.yml)
![Maitained](https://img.shields.io/badge/Maitained%3F-Yes-brightgreen)
![PyPI - Downloads](https://img.shields.io/pypi/dw/Hocrox?style=flat)
![PyPI](https://img.shields.io/pypi/v/Hocrox?style=flat)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/imdeepmind/hocrox?style=flat)
![GitHub issues](https://img.shields.io/github/issues/imdeepmind/hocrox?style=flat)
![GitHub](https://img.shields.io/github/license/imdeepmind/hocrox?style=flat)

## Introduction

Hocrox is an image preprocessing and augmentation library. It provides a [Keras](https://keras.io/) like simple interface to make preprocessing and augmentation pipelines. Hocrox internally uses [OpenCV](https://opencv.org/) to perform the operations on images. OpenCV is one of the most popular Computer Vision library.

Here are some of the highlights of Hocrox:

- Provides an easy interface that is suitable for radio pipeline development
- It internally uses OpenCV
- Highly configurable with support for custom layers

## The Keas interface

Keras is one of the most popular Deep Learning library. Keras provides a very simple yet powerful interface that can be used to develop start-of-the-art Deep Learning models.

Check the code below. This is a simple Keras code to make a simple neural network.

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

In Hocrox, the interface for making pipelines is very much similar. So anyone can make complex pipelines with few lines of code.

## Install

To install Hocrox, run the following command.

```
pip install Hocrox
```

## Dependencies

Hocrox uses OpenCV internally so install it before.

## Documentation

Documentation for Hocrox is available [here](http://hocrox.imdeepmind.com/).

## Example

Here is one simple pipeline for preprocessing images.

```python
from hocrox.model import Model
from hocrox.layer import Read, Save
from hocrox.layer.preprocessing.transformation import Resize
from hocrox.layer.augmentation.flip import RandomFlip
from hocrox.layer.augmentation.transformation import RandomRotate

# Initalizing the model
model = Model()

# Reading the images
model.add(Read(path="./images", name="Read images"))

# Resizing the images
model.add(Resize((224, 244), interpolation="INTER_LINEAR", name="Resize images"))

# Augmentating the images
model.add(
    RandomRotate(
        start_angle=-10.0, end_angle=10.0, probability=0.7, number_of_outputs=5, name="Randomly rotates the image"
    )
)
model.add(RandomFlip(probability=0.7, name="Randomly flips the image"))

# Saving the images
model.add(Save("./preprocessed_images", format="npy", name="Save the image"))

# Generating the model summary
print(model.summary())

# Transforming the images
model.transform()

```

## Contributors

Check the list of contributors [here](https://github.com/imdeepmind/hocrox/graphs/contributors).

## License

[MIT](https://github.com/imdeepmind/hocrox/blob/main/LICENSE)
