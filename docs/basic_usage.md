# Basic usage

The package defines PyTorch modules that can be used in the same way as built-in modules. For instance, the following code creates and trains a standard non-negative similarity matching (NSM) circuit:

```python linenums="1"
import torch
from torch import nn
from pynsm import SimilarityMatching

encoder = nn.Linear(784, 50)
model = SimilarityMatching(
    encoder, out_channels=50, iteration_projection=nn.RELU()
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

#
# [...]: code for loading dataset and generating loader
#

for x in train_loader:
    x = x.to(device).flatten()
    out = model(x)
    loss = model.loss(x, out)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```

The similarity matching model is defined on line 6, and it uses an encoder that must be defined separately (here it is on line 5). The encoder can be any PyTorch module.

Let us denote the encoded input by $x_\text{encoded} = W(x)$. Under the hood, this encoded input is fed into a circuit with lateral connections, whose connectivity is modeled by the matrix $M$. The lateral connectivity means that the signal is fed back into the circuit and the process is repeated until equilibrium is reached. This can be modeled by solving the ODE

$$z(\tau + d\tau) = f\bigl(z(\tau) + W(x) - M z(\tau)\bigr)\,,$$

where $f$ is a configurable non-linearity. Here we use ReLU, which was also the reason for the name "non-negative similarity matching".

The time variable in the equation above, $\tau$, refers to the time during equilibration, which is assumed to be a fast process. The full iteration until convergence of the ODE happens in the forward pass of the model (line 17). The output from the circuit is given by the value of $z$ at convergence.

On a slower timescale (i.e., in-between samples), the parameters of the encoder $W$, as well as the lateral weights $M$, are updated using a regular PyTorch optimizer (lines 18–22). The gradient necessary for the update is obtained by back-propagation based on a special loss function that can be obtained from the model itself, `model.loss(x, out)` (line 18). The ODE above is in fact obtained from the same loss function, but by doing gradient descent with respect to $z$ instead.

More detailed explanations and references regarding similarity matching can be found on the [similarity matching](similarity.md) page.
