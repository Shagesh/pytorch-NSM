# Extending the library
To extend the library, start by following the [developer installation instructions on GitHub](https://github.com/Shagesh/pytorch-NSM/blob/main/README.md#developer-installation).

## Adding a similarity-based model
You can add models by making a new file in the [`arch`](https://github.com/Shagesh/pytorch-NSM/tree/main/src/pynsm/arch) folder. Variants of similarity matching can be built by inheriting from the `MultiSimilarityMatching` class. For instance, canonical correlation analysis (CCA) can be implemented like this:[^1]

```python
from torch import nn
from pynsm import MultiSimilarityMatching

class SimilarityMatchingCCA(MultiSimilarityMatching):
    def __init__(self, dim1: int, dim2: int, out_channels: int, **kwargs):
        encoders = [
            nn.Linear(dim1, out_channels), nn.Linear(dim2, out_channels)
        ]
        super().__init__(encoders, regularization="whiten", **kwargs)
```

After adding new models, add these in `arch/__init__.py` and `__init__.py` to ensure that they can be easily accessed by users.

## Adding a more generic iteration-based model
For models where the forward pass requires iteration but are not based on similarity matching, you can instead inherit from `IterationModule` or `IterationLossModule`. The former requires that you specify the actual processing performed for every iteration, while the latter assumes that the iteration is gradient-based and only the loss function needs to be specified. In both cases, you will have to also define the state variables by overriding the `pre_iteration()` method, and define a return value by overriding the `post_iteration()` method.

Here is an example where the forward iteration generates the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set):

```python
import torch
from pynsm import IterationModule

class Mandelbrot(IterationModule):
    def pre_iteration(self, c: torch.Tensor):
        self.state = torch.zeros_like(c, dtype=torch.complex64)
        self.counts = torch.zeros_like(self.state, dtype=int)
    
    def iteration(self, c: torch.Tensor):
        self.state = self.state ** 2 + c

        mask = torch.abs(self.state) < 2
        self.counts[mask] += 1
    
    def post_iteration(self, c: torch.Tensor) -> torch.Tensor:
        self.counts = self.counts.float()
        self.counts[self.counts == self.max_iterations] = float("nan")
        return self.counts
```

We can test the code as follows:

```python
import matplotlib.pyplot as plt

extent = (-2, 0.8, -1.4, 1.4)
x = torch.linspace(extent[0], extent[1], 500)
y = torch.linspace(extent[2], extent[3], 500)
grid_real = torch.meshgrid(x, y, indexing="xy")
grid = torch.complex(*grid_real)

m = Mandelbrot(max_iterations=20)
counts = m(grid)
plt.imshow(counts, extent=extent, cmap="plasma")
```

This yields a familiar picture of the Mandelbrot set:

![Mandelbrot set](img/mandelbrot.png)


[^1]: Lipshutz, D., Bahroun, Y., Golkar, S., Sengupta, A. M., & Chklovskii, D. B. (2021). A biologically plausible neural network for multichannel canonical correlation analysis. Neural Computation, 33(9), 2309–2352. <https://doi.org/10.1162/neco_a_01414>