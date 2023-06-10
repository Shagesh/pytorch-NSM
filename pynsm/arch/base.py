"""Define base module class."""
import torch
from torch import nn

from typing import Any, Optional, Dict, List, Union, Callable


class IterationModule(nn.Module):
    """A module where the forward pass is called iteratively.

    The `forward()` method calls `self.iteration()` iteratively, until either a maximum
    number of steps is reached, or `self.converged()` is true. The current iteration
    index, as well as any arguments passed to `forward()` are passed along:

        self.iteration(idx, *args, **kwargs)

    Pre- and post-processing can be achieved by implementing `self.pre_iteration()` and
    `self.post_iteration()`, which are called before the first iteration and after the
    last, respectively. They are passed the arguments passed to `forward()`:

        self.pre_iteration(*args, **kwargs)
        self.post_iteration(*args, **kwargs)

    By default, these do nothing.
    """

    def __init__(self, max_iterations: int = 1000, **kwargs):
        """Initialize the module.

        :param max_iterations: maximum number of `iteration()` calls in one forward pass
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def forward(self, *args, **kwargs) -> Any:
        self.pre_iteration(*args, **kwargs)
        for i in range(self.max_iterations):
            self.iteration(i, *args, **kwargs)
            if self.converged(*args, **kwargs):
                break

        self.post_iteration(*args, **kwargs)

    def iteration(self, i: int, *args, **kwargs):
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "iteration" function'
        )

    def converged(self, *args, **kwargs) -> bool:
        return False

    def pre_iteration(self, *args, **kwargs):
        pass

    def post_iteration(self, *args, **kwargs):
        pass


class IterationLossModule(IterationModule):
    """A specialization of `IterationModule` where the iteration is derived from a loss
    function.

    This creates an optimizer in the `pre_iteration()`, then for each iteration runs
    `backward()` on the output from `self.iteration_loss()` and steps the optimizer. The
    constructor has options for choosing the optimizer to use, as well as an optional
    learning-rate scheduler; see below.

    Functions to implement:
      * `iteration_loss(*args, **kwargs)` should return the loss
      * `iteration_parameters()` should return a list of parameters to be optimized
        during the iteration

    Note that typically the `iteration_parameters()` should *not* be included in the
    module's `parameters()`, but should potentially be saved as part of the
    `state_dict`, so it is recommended that they be registered as buffers.
    """

    def __init__(
        self,
        iteration_optimizer: Callable = torch.optim.SGD,  # type: ignore
        iteration_scheduler: Optional[Callable] = None,
        iteration_lr: float = 1e-3,
        it_optim_kwargs: Optional[Dict[str, Any]] = None,
        it_sched_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the module.

        :param iteration_optimizer: optimizer to use for forward iteration; e.g.,
            `torch.optim.SGD`
        :param iteration_scheduler: scheduler to use (if any) for forward iteration;
            e.g., `torch.optim.lr_scheduler.StepLR`
        :param iteration_lr: learning rate for forward iteration; this is a shortcut
            that overrides any potential learning rate from `it_optim_kwargs`
        :param it_optim_kwargs: dictionary of keyword arguments to pass to the optimizer
        :param it_sched_kwargs: dictionary of keyword arguments to pass to the scheduler
        :param kwargs: other keyword arguments are passed to `IterationModule`
        """
        super().__init__(**kwargs)

        self.it_construct_optim = iteration_optimizer
        self.it_construct_sched = iteration_scheduler

        self.it_optim_kwargs = it_optim_kwargs if it_optim_kwargs is not None else {}
        self.it_optim_kwargs.setdefault("lr", iteration_lr)

        self.it_sched_kwargs = it_sched_kwargs if it_sched_kwargs is not None else {}

        self.iteration_optimizer: torch.optim.Optimizer = None  # type: ignore
        self.iteration_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

    def iteration(self, i: int, *args, **kwargs):
        self.iteration_optimizer.zero_grad()

        loss = self.iteration_loss(i, *args, **kwargs)
        loss.backward()

        self.iteration_optimizer.step()
        if self.iteration_scheduler is not None:
            self.iteration_scheduler.step()

    def pre_iteration(self, *args, **kwargs):
        super().pre_iteration(*args, **kwargs)

        for param in self.iteration_parameters():
            param.requires_grad_(True)

        self.iteration_optimizer = self.it_construct_optim(
            self.iteration_parameters(), **self.it_optim_kwargs
        )

        if self.it_construct_sched is not None:
            self.iteration_scheduler = self.it_construct_sched(
                self.iteration_optimizer, **self.it_sched_kwargs
            )

    def post_iteration(self, *args, **kwargs):
        for param in self.iteration_parameters():
            param.requires_grad_(False)

        super().post_iteration(*args, **kwargs)

    def iteration_loss(self, i: int, *args, **kwargs):
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required "
            f'"iteration_loss" function'
        )

    def iteration_parameters(self) -> List[Union[torch.Tensor, nn.Module]]:
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required "
            f'"iteration_parameters" function'
        )
