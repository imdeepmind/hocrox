"""Example code for Preprocessing images."""
from hocrox.model import Model
from hocrox.layer import Read, Save
from hocrox.layer.preprocessing.transformation import Resize
from hocrox.layer.preprocessing.color import Grayscale, Rescale

# Initalizing the model
model = Model()

# Reading the images
model.add(Read(path="./images", name="Read images"))

# Preprocessing the images
model.add(Resize((224, 244), interpolation="INTER_LINEAR", name="Resize images"))
model.add(Grayscale(name="Grayscaled images"))
model.add(Rescale(rescale=1 / 255, name="Normalize images"))

# Saving the images
model.add(Save("./preprocessed_images", format="npy", name="Save the image"))

# Generating the model summary
print(model.summary())

# Here is the model summary for reference
# +-------+------------------------------+-----------------------------------+
# | Index |             Name             |             Parameters            |
# +-------+------------------------------+-----------------------------------+
# |   #1  |      Read images(read)       |            Path: ./img            |
# |   #2  |    Resize images(resize)     | Dim: (224, 244), Interpolation: 1 |
# |   #3  | Grayscaled images(greyscale) |                 -                 |
# |   #4  |  Normalize images(rescale)   |    Rescale: 0.00392156862745098   |
# |   #5  |     Save the image(save)     |     Path: ./img2, Format: npy     |
# +-------+------------------------------+-----------------------------------+

# Transforming the images
model.transform()
