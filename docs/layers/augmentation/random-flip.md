# Random Flip

Randomly flips the image vertically or horizontally.

```python
hocrox.layers.augmentation.RandomFlip(number_of_outputs=1, name=None)
```

## Supported Arguments:

- `number_of_outputs`: (Integer) Number of images to output
- `name=None`: (String) Name of the layer, if not provided then automatically generates an unique name for the layer

## Example Code:

```python
from hocrox.model import Model
from hocrox.layer.augmentation import RandomFlip

# Initializing the model
model = Model("./img")

# Adding model layers
model.add(RandomFlip(number_of_outputs=1))

# Printing the summary of the model
print(model.summary())
```
