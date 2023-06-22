"""Define a convolutional NSM module."""

import math

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt

from typing import List, Optional, Union

from .base import IterationLossModule


class SimilarityMatching(IterationLossModule):
    """Similarity matching circuit."""

    def __init__(
        self,
        encoder: nn.Module,
        out_channels: int,
        tau: float = 0.1,
        max_iterations: int = 40,
        regularization: str = "weight",
        **kwargs,
    ):
        """Initialize the similarity matching circuit.

        :param encoder: module to use for encoding the inputs
        :param out_channels: number of output channels
        :param tau: factor by which to divide the competitor's learning rate
        :param max_iterations: maximum number of iterations to run in `forward()`
        :param regularization: type of encoder regularization to use; this can be
            "weight":   use the encoder's parameters; an exception is raised in the
                        ambiguous case when`encoder.parameters()` has length > 1
            "whiten":   use a regularizer that encourages whitening XXX explain
            "none":     do not use regularization for the encoder; most useful to allow
                        for custom regularization, since lack of regularization leads to
                        unstable dynamics in many cases
        :param **kwargs: additional keyword arguments passed to `IterationLossModule`
        """
        super().__init__(max_iterations=max_iterations, **kwargs)

        self.encoder = encoder
        self.out_channels = out_channels
        self.tau = tau
        self.regularization = regularization

        if self.regularization not in ["weight", "whiten", "none"]:
            raise ValueError(f"Unknown regularization {self.regularization}")

        # XXX should probably ensure the device (and data types?) match
        self.competitor = nn.Linear(out_channels, out_channels, bias=False)
        torch.nn.init.eye_(self.competitor.weight)

        # make sure we maximize with respect to competitor weight...
        # ...and implement the learning rate ratio
        scaling = -1.0 / tau
        self.competitor.weight.register_hook(lambda g: g * scaling)

        self.y = torch.tensor([])

    def pre_iteration(self, x: torch.Tensor):
        self._Wx = self.encoder(x).detach()
        self.y = torch.zeros_like(self._Wx)
        super().pre_iteration(x)

    def iteration_set_gradients(self, x: torch.Tensor):
        with torch.no_grad():
            My = torch.einsum("ij,bj... -> bi...", self.competitor.weight, self.y)
            self.y.grad = My - self._Wx

    def iteration_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Loss function associated with the iteration.

        This is not actually used by the iteration, which instead uses manually
        calculated gradients (for efficiency).
        """
        assert self._Wx is not None
        loss = self._loss_no_reg(self._Wx, self.y, "sum")
        return loss / 4

    def post_iteration(self, x: torch.Tensor):
        super().post_iteration(x)
        self._Wx = None

        return self.y.detach()

    def iteration_parameters(self) -> List[torch.Tensor]:
        return [self.y]

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        Wx = self.encoder(x)
        loss = self._loss_no_reg(Wx, y, "mean")

        # competitor regularization
        M_reg = (self.competitor.weight**2).sum() / y.shape[1]
        loss -= M_reg

        # encoder regularization
        if self.regularization == "whiten":
            # this needs to match the scaling from _loss_no_reg!
            loss += 2 * (Wx**2).mean()
        elif self.regularization == "weight":
            encoder_params = list(self.encoder.parameters())
            if len(encoder_params) == 0:
                raise ValueError(
                    "Cannot use weight regularizer because encoder has no parameters"
                )
            if len(encoder_params) > 1:
                raise ValueError(
                    "Ambiguity in weight regularizer as encoder has more "
                    "than one parameter tensor"
                )
            weight = encoder_params[0]
            loss += (weight**2).sum() * (2.0 / y.shape[1])
        elif self.regularization != "none":
            raise ValueError(f"Unknown regularization {self.regularization}")

        return loss

    def _loss_no_reg(self, Wx: torch.Tensor, y: torch.Tensor, reduction: str):
        """Compute the part of the loss without the regularization terms.

        :param Wx: encoded input, `self.encoder(x)`
        :param y: output (after iteration converges)
        :param reduction: "mean" or "sum"
        """
        My = torch.einsum("ij,bj... -> bi...", self.competitor.weight, y)
        yWx = (y * Wx).sum()
        yMy = (y * My).sum()

        loss = -4 * yWx + 2 * yMy

        if reduction == "mean":
            loss /= torch.numel(y)

        return loss
