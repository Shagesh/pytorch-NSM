import pytest

import torch
from torch import nn

from typing import List

from unittest.mock import Mock

from pynsm.arch import IterationModule, IterationLossModule


class MockModule(IterationModule):
    def __init__(self, starts_converged: bool, max_iterations: int = 1000):
        super().__init__(max_iterations=max_iterations)
        self.is_converged = starts_converged
        self.n_calls = 0

    def iteration(self, *args, **kwargs):
        self.n_calls += 1

    def converged(self) -> bool:
        return self.is_converged


class MockModuleWithPrePost(IterationModule):
    def __init__(self):
        super().__init__(max_iterations=5)
        self.n_calls_pre = 0
        self.n_calls_post = 0

        self.last_call = None
        self.last_pre_call = None
        self.last_post_call = None
        self.last_conv_call = None

    def iteration(self, *args, **kwargs):
        self.last_call = (args, kwargs)

    def pre_iteration(self, *args, **kwargs):
        self.n_calls_pre += 1
        self.last_pre_call = (args, kwargs)

    def post_iteration(self, *args, **kwargs) -> str:
        self.n_calls_post += 1
        self.last_post_call = (args, kwargs)
        return "foobar"

    def converged(self, *args, **kwargs) -> bool:
        self.last_conv_call = (args, kwargs)
        return False


class MockLossModule(IterationLossModule):
    def __init__(self, n: int = 5, **kwargs):
        super().__init__(**kwargs)

        torch.manual_seed(42)
        self.n = n
        self.decoy = nn.Linear(2, 3, bias=False)
        self.register_buffer("state", torch.randn(self.n))
        self.loss_history = []

    def iteration_loss(self, *args, **kwargs) -> torch.Tensor:
        loss = ((self.state + torch.ones(self.n)) ** 2).sum()
        self.loss_history.append(loss.item())
        return loss

    def iteration_parameters(self) -> List[torch.Tensor]:
        return [self.state]  # type: ignore

    def post_iteration(self, *args, **kwargs) -> str:
        super().post_iteration(*args, **kwargs)
        return "foobar"


def test_base_inherits_from_module():
    module = IterationModule()
    assert isinstance(module, nn.Module)


def test_forward_on_base_raises_not_implemented():
    module = IterationModule()
    with pytest.raises(NotImplementedError):
        module()


def test_iteration_called_until_max_iterations():
    n = 35
    module = MockModule(False, max_iterations=n)

    assert module.n_calls == 0
    module()
    assert module.n_calls == n


def test_iteration_ends_when_converged_is_true():
    module = MockModule(True)

    assert module.n_calls == 0
    module()
    assert module.n_calls == 1


def test_pre_and_post_iteration_are_called_once():
    module = MockModuleWithPrePost()

    assert module.n_calls_pre == 0
    assert module.n_calls_post == 0
    module()
    assert module.n_calls_pre == 1
    assert module.n_calls_post == 1


def test_args_kwargs_passed_to_iteration_and_pre_post_converged():
    module = MockModuleWithPrePost()
    module(2, 3, foo="bar")

    for last in [
        module.last_call,
        module.last_pre_call,
        module.last_post_call,
        module.last_conv_call,
    ]:
        assert last == ((2, 3), {"foo": "bar"})


def test_forward_returns_output_from_post_iteration():
    module = MockModuleWithPrePost()
    ret = module(2)

    assert ret == "foobar"


def test_pre_hook_called_before_pre_iteration():
    module = MockModuleWithPrePost()
    hook_called = [False]

    def hook(m):
        hook_called[0] = True
        assert m.last_pre_call is None
        assert m.last_call is None

    module.register_iteration_hook("pre", hook)
    module()

    assert hook_called[0]


def test_post_hook_called_after_post_iteration():
    module = MockModuleWithPrePost()
    hook_called = [False]

    def hook(m):
        hook_called[0] = True
        assert m.last_post_call is not None

    module.register_iteration_hook("post", hook)
    module()

    assert hook_called[0]


def test_iteration_hook_called_after_every_iteration():
    module = MockModuleWithPrePost()
    n_calls = [0]

    def hook(m):
        n_calls[0] += 1
        assert m.last_call is not None

    module.register_iteration_hook("iteration", hook)
    module()

    assert n_calls[0] == module.max_iterations


def test_register_multiple_hooks():
    module = MockModuleWithPrePost()
    hooks_called = [False, False]

    def hook1(m):
        hooks_called[0] = True

    def hook2(m):
        hooks_called[1] = True

    module.register_iteration_hook("iteration", hook1)
    module.register_iteration_hook("iteration", hook2)
    module()

    assert all(hooks_called)


def test_iteration_stops_if_iteration_hook_returns_truthful():
    module = MockModuleWithPrePost()
    n_calls = [0]

    def hook(m):
        n_calls[0] += 1
        return True

    module.register_iteration_hook("iteration", hook)
    module()

    assert n_calls[0] == 1


def test_iteration_idx_updated():
    n = 13
    module = MockModule(False, max_iterations=n)

    idxs = []
    module.register_iteration_hook("iteration", lambda m: idxs.append(m.iteration_idx))
    module()

    assert idxs == list(range(n))


def test_loss_of_base_loss_model_raises_not_implemented():
    module = IterationLossModule()
    with pytest.raises(NotImplementedError):
        module.iteration_loss()


def test_loss_model_forward_lowers_iteration_loss():
    module = MockLossModule()

    loss0 = module.iteration_loss().item()
    module()
    loss1 = module.iteration_loss().item()

    assert loss1 < loss0


def test_loss_model_forward_resets_iter_params_grad_state_to_false():
    module = MockLossModule()
    module()
    assert not module.state.requires_grad


def test_loss_model_change_optimizer():
    mock_optim = Mock()
    mock_optim_class = Mock(return_value=mock_optim)
    module = MockLossModule(iteration_optimizer=mock_optim_class)
    module()

    mock_optim_class.assert_called_once()
    mock_optim.zero_grad.assert_called()
    mock_optim.step.assert_called()


def test_loss_model_use_scheduler():
    mock_sched = Mock()
    mock_sched_class = Mock(return_value=mock_sched)
    module = MockLossModule(iteration_scheduler=mock_sched_class)
    module()

    mock_sched_class.assert_called_once()
    mock_sched.step.assert_called()


def test_loss_model_it_optim_kwargs_passed_to_optimizer():
    mock_optim = Mock()
    module = MockLossModule(
        iteration_optimizer=mock_optim, it_optim_kwargs={"foo": "bar"}
    )
    module()

    assert "foo" in mock_optim.call_args.kwargs
    assert mock_optim.call_args.kwargs["foo"] == "bar"


def test_loss_model_it_sched_kwargs_passed_to_scheduler():
    mock_sched = Mock()
    module = MockLossModule(
        iteration_scheduler=mock_sched, it_sched_kwargs={"foo": "bar"}
    )
    module()

    assert "foo" in mock_sched.call_args.kwargs
    assert mock_sched.call_args.kwargs["foo"] == "bar"


def test_loss_model_projection():
    # make sure we would have some negative elements without a projection
    module1 = MockLossModule(n=3)
    module1()
    assert torch.any(module1.state < 0)  # type: ignore

    module2 = MockLossModule(n=3, iteration_projection=nn.functional.relu)
    module2()
    assert torch.all(module2.state >= 0)  # type: ignore


def test_loss_model_projection_after_every_iteration():
    # make sure we would have some negative elements without a projection
    module1 = MockLossModule(n=3)
    module1()
    assert torch.any(module1.state < 0)  # type: ignore

    module2 = MockLossModule(n=3, iteration_projection=nn.functional.relu)
    module2.pre_iteration()
    module2.iteration()
    assert torch.all(module2.state >= 0)  # type: ignore


def test_pre_post_iteration_toggles_requires_grad_for_it_params():
    module = MockLossModule()
    module.pre_iteration()
    assert module.state.requires_grad

    module.post_iteration()
    assert not module.state.requires_grad


def test_pre_post_iteration_toggles_requires_grad_for_params():
    module = MockLossModule()
    params = list(module.parameters())
    assert len(params) > 0
    assert params[0].requires_grad

    module.pre_iteration()
    assert not params[0].requires_grad

    module.post_iteration()
    assert params[0].requires_grad


def test_loss_model_forward_returns_output_from_post_iteration():
    module = MockLossModule()
    ret = module(2)

    assert ret == "foobar"


def test_loss_model_updates_iteration_loss():
    module = MockLossModule()
    loss_history = []

    module.register_iteration_hook(
        "iteration", lambda m: loss_history.append(m.last_iteration_loss)
    )
    module()

    assert pytest.approx(loss_history) == module.loss_history
