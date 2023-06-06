"""Define base module class."""
from torch import nn

from typing import Any


class IterationModule(nn.Module):
    """A module where the forward pass is called iteratively.

    The `forward()` method calls `self.iteration()` iteratively, until either a maximum
    number of steps is reached, or `self.converged()` is true. The current iteration
    index is passed as an argument.

    Pre- and post-processing can be achieved by implementing `self.pre_iteration()` and
    `self.post_iteration()`, which are called before the first iteration and after the
    last, respectively. By default, these do nothing.
    """

    def __init__(self, *args, max_iterations: int = 1000, **kwargs):
        """Initialize the module.

        :param max_iterations: maximum number of `iteration()` calls in one forward pass
        """
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations

    def forward(self, *args) -> Any:
        self.pre_iteration(*args)
        for i in range(self.max_iterations):
            self.iteration(i)
            if self.converged():
                break

        self.post_iteration()

    def iteration(self, i: int):
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "iteration" function'
        )

    def converged(self) -> bool:
        return False

    def pre_iteration(self, *args):
        pass

    def post_iteration(self, *args):
        pass
