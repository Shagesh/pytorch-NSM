"""Define base module class."""
import torch
from torch import nn

from collections import defaultdict
from typing import Any, Optional, Dict, List, Union, Callable


class IterationModule(nn.Module):
    """A module where the forward pass is called iteratively.

    The `forward()` method calls `self.iteration()` iteratively, until either a maximum
    number of steps is reached, or `self.converged()` is true. Any arguments passed to
    `forward()` are passed along:

        self.iteration(*args, **kwargs)

    When done iterating, `forward()` returns the output from `self.post_iteration()`.
    The methods `self.pre_iteration()` and `self.post_iteration()` can also be used to
    perform any necessary pre- and post-processing, as they are called before the first
    iteration and after the last, respectively. They are passed the arguments passed to
    `forward()`:

        self.pre_iteration(*args, **kwargs)
        self.post_iteration(*args, **kwargs)

    By default, these do nothing and return nothing.
    """

    def __init__(self, max_iterations: int = 1000, **kwargs):
        """Initialize the module.

        :param max_iterations: maximum number of `iteration()` calls in one forward pass
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.hooks = defaultdict(list)
        self.iteration_idx = -1

    def register_iteration_hook(self, kind: str, hook: Callable):
        """Register a hook for the forward iteration.

        Hooks can be assigned for the following events:
        * pre:          called before `pre_iteration()`
            signature: `hook(module)`
        * post:         called after `post_iteration()`
            signature: `hook(module)`
        * iteration:    called after every call to `iteration()` (before `converged()`)
            signature: `hook(module) -> bool`
            A truthful return value ends the iteration. The iteration index is available
            in `module.iteration_idx`.

        Multiple hooks can be attached to the same event.

        :param kind: the kind of hook to register
        :param hook: the function to be called
        """
        if kind not in ["pre", "post", "iteration"]:
            raise ValueError(f"Unknown event type in register_iteration_hook: {kind}")

        self.hooks[kind].append(hook)

    def forward(self, *args, **kwargs) -> Any:
        self.iteration_idx = -1

        self._call_hooks(self.hooks["pre"])
        self.pre_iteration(*args, **kwargs)
        for i in range(self.max_iterations):
            self.iteration_idx = i
            self.iteration(*args, **kwargs)
            if self._call_hooks(self.hooks["iteration"], check_output=True):
                break
            if self.converged(*args, **kwargs):
                break

        retval = self.post_iteration(*args, **kwargs)
        self._call_hooks(self.hooks["post"])

        return retval

    def iteration(self, *args, **kwargs):
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "iteration" function'
        )

    def converged(self, *args, **kwargs) -> bool:
        return False

    def pre_iteration(self, *args, **kwargs):
        pass

    def post_iteration(self, *args, **kwargs) -> Any:
        pass

    def _call_hooks(self, hooks: List[Callable], check_output: bool = False) -> bool:
        for hook in hooks:
            retval = hook(self)
            if check_output and retval:
                return True

        return False


class IterationLossModule(IterationModule):
    """A specialization of `IterationModule` where the iteration is derived from a loss
    function.

    This creates an optimizer in the `pre_iteration()`, then for each iteration runs
    `backward()` on the output from `self.iteration_loss()` and steps the optimizer.
    This is followed by an optional projection step; see `iteration_projection` below.
    Note that projection is a very simple way of enforcing constraints, and might not
    work well with adaptive step optimizers.

    The constructor has options for choosing the optimizer to use, as well as for an
    optional learning-rate scheduler; see below.

    Functions to implement:
      * `iteration_loss(*args, **kwargs)` should return the loss; the output is stored
        in `self.last_iteration_loss` as a number (i.e. `item()` is called on the tensor
        output from `iteration_loss()`)
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
        iteration_projection: Optional[Callable] = None,
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
        :param iteration_projection: optional projection to perform after each optimizer
            step; this should be a callable that will be applied to each element of
            `self.iteration_parameters()` (e.g., `torch.nn.functional.relu`)
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

        self.iteration_projection = iteration_projection

        self.last_iteration_loss = None

    def iteration(self, *args, **kwargs):
        self.iteration_optimizer.zero_grad()

        loss = self.iteration_loss(*args, **kwargs)
        loss.backward()

        self.iteration_optimizer.step()
        if self.iteration_scheduler is not None:
            self.iteration_scheduler.step()

        if self.iteration_projection is not None:
            with torch.no_grad():
                for param in self.iteration_parameters():
                    param.data = self.iteration_projection(param.data)

        self.last_iteration_loss = loss.item()

    def pre_iteration(self, *args, **kwargs):
        """Pre-iteration processing.

        This sets `requires_grad` to `True` for the `iteration_parameters()` and to
        `False` for the `parameters()`. It also generates an optimizer and a scheduler,
        if one is requested.
        """
        super().pre_iteration(*args, **kwargs)

        # iteration parameters require grad...
        for param in self.iteration_parameters():
            param.requires_grad_(True)
        # ...but other parameters don't
        for param in self.parameters():
            param.requires_grad_(False)

        self.iteration_optimizer = self.it_construct_optim(
            self.iteration_parameters(), **self.it_optim_kwargs
        )

        if self.it_construct_sched is not None:
            self.iteration_scheduler = self.it_construct_sched(
                self.iteration_optimizer, **self.it_sched_kwargs
            )

    def post_iteration(self, *args, **kwargs) -> Any:
        """Post-iteration processing.

        This sets `requires_grad` to `False` for the `iteration_parameters()` and to
        `True` for the `parameters()`. Note that the state of these parameters before
        `pre_iteration()` is not checked.
        """
        # reset: no grad for iteration parameters...
        for param in self.iteration_parameters():
            param.requires_grad_(False)
        # ...and yes grad for the others
        for param in self.parameters():
            param.requires_grad_(True)

        return super().post_iteration(*args, **kwargs)

    def iteration_loss(self, *args, **kwargs):
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required "
            f'"iteration_loss" function'
        )

    def iteration_parameters(self) -> List[Union[torch.Tensor, nn.Module]]:
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required "
            f'"iteration_parameters" function'
        )
