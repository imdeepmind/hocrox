# Random Rotate

Randomly rotates the image to certain angle.

```python
hocrox.layers.augmentation.RandomRotate(start_angle, end_angle, number_of_outputs=1, name=None)
```

## Supported Arguments:

- `start_angle`: (Float) Starting range for the angle
- `end_angle`: (Float) Ending range for the angle
- `number_of_outputs`: (Integer) Number of images to output
- `name=None`: (String) Name of the layer, if not provided then automatically generates an unique name for the layer

## Example Code:

```python
from hocrox.model import Model
from hocrox.layer.augmentation import RandomRotate

# Initializing the model
model = Model("./img")

# Adding model layers
model.add(RandomRotate(start_angle=-10.0, end_angle=10.0, number_of_outputs=5))

# Printing the summary of the model
print(model.summary())
```
