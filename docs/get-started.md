---
title: Get Started
description: Get Started with Hocrox library. Find all the important instructions to start using Hocrox.
authors:
  - Abhishek Chatterjee
---

# Get Started

Here, in this example, we’ll create a simple preprocessing and augmentation model.

## Dataset for the model

For the dataset, just collect some images from the internet and put it in one folder.

## Install the library

Check the [install](/install/) page for installation instructions.

## Importing dependencies from Hocrox

Let’s import the dependencies from Hocrox.

Here we make a simple model with some basic preprocessing and augmentation layers.

```python
from hocrox.model import Model

from hocrox.layer.preprocessing.transformation import Grayscale, Resize
from hocrox.layer.augmentation.flip import RandomFlip
from hocrox.layer.augmentation.transformation import RandomRotate
from hocrox.layer import Read, Save
```

## Making the model#

The model class provides an easy `.add()` method to add layers. We will use the .add() method here to add some layers.

```python
# Initializing the model
model = Model()

# Adding model layers
model.add(Read(path="./img"))
model.add(Resize((224, 224)))
model.add(Grayscale())
model.add(RandomFlip(probability=1.0, number_of_outputs=2))
model.add(RandomRotate(probability=1.0, start_angle=-10, end_angle=10, number_of_outputs=5))
model.add(Save("./processed_images"))
```

## Summary of the model

Once we defined the model, it is a good idea to print the model summary to make sure the pipeline is correct.

To print summary, we have a simple `.summary()` method. We will use it here.

```python
# Printing the summary of the model
print(model.summary())
```

## Transforming the images

Once we are done with the model pipeline, we can use the `.transform()` method transform the images based on the model pipeline.

The transform method transform the images and saves it into the given path.

```python
# Apply transform to the images
model.transform()
```
