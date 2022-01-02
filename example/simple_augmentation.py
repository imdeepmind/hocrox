"""Example code for Augmentating images."""
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

# Here is the model summary for reference
# +-------+-------------------------------------------+----------------------------------------+
# | Index |                    Name                   |               Parameters               |
# +-------+-------------------------------------------+----------------------------------------+
# |   #1  |             Read images(read)             |              Path: ./img               |
# |   #2  |           Resize images(resize)           |   Dim: (224, 244), Interpolation: 1    |
# |   #3  | Randomly rotates the image(random_rotate) | Probability: 0.7, Number of Outputs: 5 |
# |   #4  |   Randomly flips the image(random_flip)   | Probability: 0.7, Number of Outputs: 1 |
# |   #5  |            Save the image(save)           |       Path: ./img2, Format: npy        |
# +-------+-------------------------------------------+----------------------------------------+

# Transforming the images
model.transform()
