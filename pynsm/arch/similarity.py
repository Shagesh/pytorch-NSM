"""Define a convolutional NSM module."""

import torch
from torch import nn

from typing import List, Sequence, Tuple, Optional

from .base import IterationLossModule


class MultiSimilarityMatching(IterationLossModule):
    """Multiple-target similarity matching circuit."""

    def __init__(
        self,
        encoders: Sequence[nn.Module],
        out_channels: int,
        tau: float = 0.1,
        max_iterations: int = 40,
        regularization: str = "weight",
        **kwargs,
    ):
        """Initialize the supervised similarity matching circuit.

        Some of the encoders can be skipped during the `forward()` call either by
        including fewer arguments than `len(encoders)` or by setting some to `None`.

        :param encoders: modules to use for encoding the inputs
        :param out_channels: number of output channels
        :param tau: factor by which to divide the competitor's learning rate
        :param max_iterations: maximum number of iterations to run in `forward()`
        :param regularization: type of encoder regularization to use; this can be
            "weight":   use the encoders' parameters; regularization is added for all
                        the tensors returned by `encoder.parameters()` for each
                        `encoder`, as long as those tensors are trainable (i.e.,
                        `requires_grad` is true)
            "whiten":   use a regularizer that encourages whitening XXX explain
            "none":     do not use regularization for the encoder; most useful to allow
                        for custom regularization, since lack of regularization leads to
                        unstable dynamics in many cases
        :param **kwargs: additional keyword arguments passed to `IterationLossModule`
        """
        super().__init__(max_iterations=max_iterations, **kwargs)

        self.encoders = nn.ModuleList(encoders)
        self.out_channels = out_channels
        self.tau = tau
        self.regularization = regularization

        if self.regularization not in ["weight", "whiten", "none"]:
            raise ValueError(f"Unknown regularization {self.regularization}")

        self.competitor = nn.Linear(out_channels, out_channels, bias=False)
        torch.nn.init.eye_(self.competitor.weight)

        # make sure we maximize with respect to competitor weight...
        # ...and implement the learning rate ratio
        scaling = -1.0 / tau
        self.competitor.weight.register_hook(lambda g: g * scaling)

        self.y = torch.tensor([])

    def _encode(self, *args: Optional[torch.Tensor]) -> Sequence[torch.Tensor]:
        Wx = []
        for x, encoder in zip(args, self.encoders):
            if x is not None:
                Wx.append(encoder(x))

        return Wx

    def pre_iteration(self, *args: Optional[torch.Tensor]):
        assert len(args) == len(self.encoders)

        Wx = self._encode(*args)
        self._Wx = [_.detach() for _ in Wx]

        Wx_sum = self._Wx[0]
        for w in self._Wx[1:]:
            Wx_sum += w
        self._Wx_sum = Wx_sum

        self.y = torch.zeros_like(Wx[0])
        super().pre_iteration(*args)

    def iteration_set_gradients(self, *args: Optional[torch.Tensor]):
        with torch.no_grad():
            My = torch.einsum("ij,bj... -> bi...", self.competitor.weight, self.y)
            self.y.grad = My - self._Wx_sum

    def iteration_loss(self, *args: Optional[torch.Tensor]) -> torch.Tensor:
        """Loss function associated with the iteration.

        This is not actually used by the iteration, which instead uses manually
        calculated gradients (for efficiency).
        """
        assert self._Wx is not None
        loss = self._loss_no_reg(self._Wx, self.y, "sum")
        return loss / 4

    def post_iteration(self, *args: Optional[torch.Tensor]):
        super().post_iteration(*args)
        self._Wx = None
        self._Wx_sum = None

        return self.y.detach()

    def iteration_parameters(self) -> List[torch.Tensor]:
        return [self.y]

    def loss(self, *args: Optional[torch.Tensor]):
        y = args[-1]
        args = args[:-1]

        assert y is not None

        Wx = self._encode(*args)
        loss = self._loss_no_reg(Wx, y, "mean")

        # competitor regularization
        M_reg = (self.competitor.weight**2).sum() / y.shape[1]
        loss -= M_reg

        # encoder regularization
        if self.regularization == "whiten":
            # this needs to match the scaling from _loss_no_reg!
            for crt_Wx in Wx:
                loss += 2 * (crt_Wx**2).mean()
        elif self.regularization == "weight":
            for encoder in self.encoders:
                encoder_params = [_ for _ in encoder.parameters() if _.requires_grad]
                for weight in encoder_params:
                    loss += (weight**2).sum() * (2.0 / y.shape[1])
        elif self.regularization != "none":
            raise ValueError(f"Unknown regularization {self.regularization}")

        return loss

    def _loss_no_reg(
        self, Wx: Sequence[torch.Tensor], y: torch.Tensor, reduction: str
    ) -> torch.Tensor:
        """Compute the part of the loss without the regularization terms.

        :param Wx: encoded input, `self.encoder(x)`
        :param y: output (after iteration converges)
        :param reduction: "mean" or "sum"
        """
        My = torch.einsum("ij,bj... -> bi...", self.competitor.weight, y)
        yMy = (y * My).sum()

        loss = 2 * yMy

        for crt_Wx in Wx:
            crt_yWx = (y * crt_Wx).sum()
            loss -= 4 * crt_yWx

        if reduction == "mean":
            loss /= torch.numel(y)

        return loss


class SimilarityMatching(MultiSimilarityMatching):
    """Similarity matching circuit."""

    def __init__(self, encoder: nn.Module, *args, **kwargs):
        """Initialize the similarity matching circuit.

        This is a thin wrapper around `MultiSimilarityMatching` using a single target.

        :param encoder: module to use for encoding the inputs
        :param *args: additional positional arguments go to `MultiSimilarityMatching`
        :param **kwargs: additional keyword arguments go to `MultiSimilarityMatching`
        """
        super().__init__(encoders=[encoder], *args, **kwargs)
