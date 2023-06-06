import pytest

from torch import nn

from pynsm.arch import IterationModule


class MockModule(IterationModule):
    def __init__(self, starts_converged: bool, max_iterations: int = 1000):
        super().__init__(max_iterations=max_iterations)
        self.is_converged = starts_converged
        self.n_calls = 0

    def iteration(self, i: int):
        assert i == self.n_calls
        self.n_calls += 1

    def converged(self) -> bool:
        return self.is_converged


class MockModuleWithPrePost(IterationModule):
    def __init__(self):
        super().__init__()
        self.n_calls_pre = 0
        self.n_calls_post = 0

    def iteration(self, i: int):
        pass

    def pre_iteration(self):
        self.n_calls_pre += 1

    def post_iteration(self):
        self.n_calls_post += 1


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
