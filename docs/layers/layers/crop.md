# Crop

Crops the image.

```python
hocrox.layers.preprocessing.Crop(x, y, w, h, name=None)
```

## Supported Arguments:

- `x`: (Integer) X coordinate for the crop
- `y`: (Integer) Y coordinate for the crop
- `w`: (Integer) Width of the box
- `h`: (Integer) Height of the box
- `number_of_outputs`: (Integer) Number of images to output
- `name=None`: (String) Name of the layer, if not provided then automatically generates an unique name for the layer

## Example Code:

```python
from hocrox.model import Model
from hocrox.layer.preprocessing import Crop

# Initializing the model
model = Model("./img")

# Adding model layers
model.add(Crop(x=10, y=10, w=100, h=100))

# Printing the summary of the model
print(model.summary())
```
