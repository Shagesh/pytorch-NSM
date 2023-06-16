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

        self.y: torch.Tensor
        self.register_buffer("y", torch.tensor([]))

    def pre_iteration(self, x: torch.Tensor):
        self._Wx = self.encoder(x).detach()
        self.y = torch.zeros_like(self._Wx)
        super().pre_iteration(x)

    def iteration_loss(self, x: torch.Tensor):
        assert self._Wx is not None
        return self._loss_no_reg(self._Wx, self.y)

    def post_iteration(self, x: torch.Tensor):
        super().post_iteration(x)
        self._Wx = None

        return self.y.detach()

    def iteration_parameters(self) -> List[torch.Tensor]:
        return [self.y]

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        Wx = self.encoder(x)
        loss = self._loss_no_reg(Wx, y)

        # competitor regularization
        M_reg = (self.competitor.weight**2).sum()
        loss -= 0.5 * M_reg

        # encoder regularization
        if self.regularization == "whiten":
            # this needs to match the scaling from _loss_no_reg!
            scaling_factor = y.shape[1]
            loss += (Wx**2).mean() * scaling_factor
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
            loss += (weight**2).sum()
        elif self.regularization != "none":
            raise ValueError(f"Unknown regularization {self.regularization}")

        return loss

    def _loss_no_reg(self, Wx: torch.Tensor, y: torch.Tensor):
        """Compute the part of the loss without the regularization terms.

        :param Wx: encoded input, `self.encoder(x)`
        :param y: output (after iteration converges)
        """
        scaling_factor = y.shape[1]
        yWx = (y * Wx).mean() * scaling_factor

        My = torch.einsum("ij,bj... -> bi...", self.competitor.weight, y)
        yMy = (y * My).mean() * scaling_factor

        loss = -2 * yWx + yMy
        return loss


class NSM_Conv(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        k_size,
        etaW=1e-3,
        etaM=1e-3,
        tau=1e-1,
        dropout_p=None,
        x_whitening=False,
    ):
        super(NSM_Conv, self).__init__()
        ## encoder ##
        self.in_channels = input_dim
        self.encoding_dim = encoding_dim
        self.encoder = nn.Conv2d(input_dim, encoding_dim, k_size, stride=1, padding=0)
        torch.nn.init.normal_(self.encoder.weight, mean=0.0, std=1.0)
        for iter_norm in range(encoding_dim):
            self.encoder.weight[iter_norm].data /= torch.linalg.norm(
                self.encoder.weight[iter_norm].detach()
            )

        ## competitor ##
        self.competitor = nn.Linear(encoding_dim, encoding_dim, bias=False)
        self.competitor.weight.data.copy_(torch.eye(encoding_dim))  # /encoding_dim)
        self.dropout_p = dropout_p

        ## dropout and input whitening ##
        if self.dropout_p is not None:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.x_whitening = x_whitening

        ## learning rates
        self.etaW = etaW
        self.etaM = etaM
        self.tau = tau

        # Decay/schedule
        self.t_step = 0
        self.decay = 1e-1

    def forward(self, x):
        num_iter = 40
        Wx = self.encoder(x).detach()
        if self.dropout_p is not None:
            Wx = (1 / self.dropout_p) * self.dropout(Wx)

        device = x.device
        M = self.competitor.weight.detach() - torch.eye(
            self.competitor.weight.detach().size(0)
        ).to(device)

        u = torch.zeros(Wx.size()).to(device)
        y = torch.zeros(Wx.size()).to(device)

        for iter_updates in range(num_iter):
            My = torch.einsum("ij,kjxy->kixy", M, y)
            # delta_u = - u + Wx - My
            u = Wx - My
            y = F.relu(u)

        # if self.dropout is not None:
        #  y = self.dropout(y)

        return y.detach()

    def pool_output(self, y, k_size, stri):
        m = nn.MaxPool2d(kernel_size=k_size, stride=stri)
        return m(y).detach()

    def pool_quadrant(self, y):
        Width_Im = y.shape[2]

        if Width_Im % 2 == 0:
            kernel_size = Width_Im // 2
            stride = kernel_size
        else:
            kernel_size = math.ceil(Width_Im / 2)
            stride = kernel_size - 1

        m = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        return m(y).detach()

    def loss_NSM_conv(self, y, x, print_=False):
        scaling_factor = y.shape[0] * y.shape[2] * y.shape[3]

        Wx = self.encoder(x)
        YWX = (y * Wx).sum() / scaling_factor

        My = torch.einsum("ij,kjxy->kixy", self.competitor.weight, y)
        YMY = (y * My).sum() / scaling_factor

        if self.x_whitening == True:
            W_reg = (Wx**2).sum() / scaling_factor
        else:
            W_reg = (self.encoder.weight**2).sum()

        M_reg = (self.competitor.weight**2).sum()
        total_loss = -4 * YWX + 2 * YMY + 2 * W_reg - M_reg
        if print_ == True:
            print(
                W_reg.detach().cpu(),
                M_reg.detach().cpu(),
                YWX.detach().cpu(),
                YMY.detach().cpu(),
            )
        return total_loss

    def train(self, y, x, etaW=None, etaM=None, tau=None):
        if etaW is None:
            etaW = self.etaW
        if etaM is None:
            etaM = self.etaM
        if tau is None:
            tau = self.tau

        self.loss_NSM_conv(y, x).backward()

        # gradients
        gW: torch.Tensor = self.encoder.weight.grad  # type: ignore
        gM: torch.Tensor = self.competitor.weight.grad  # type: ignore

        # updates
        lr_schedule = 1  # 1 / ( 1 + self.t_step * self.decay )

        with torch.no_grad():
            self.encoder.weight -= etaW * lr_schedule * gW
            self.competitor.weight += (etaM / tau) * lr_schedule * gM

        self.encoder.weight.grad.zero_()  # type: ignore
        self.competitor.weight.grad.zero_()  # type: ignore
        self.t_step += 10

    def plot_features(self):
        # function to plot features

        if self.in_channels == 1:
            k_size = self.encoder.kernel_size[0]
            plt.figure(figsize=(10, 10))
            for iter_filter in range(min(25, self.encoding_dim)):
                plt.subplot(5, 5, iter_filter + 1)
                plt.imshow(
                    self.encoder.weight[iter_filter]
                    .reshape(self.in_channels, k_size, k_size)
                    .detach()
                    .cpu()
                )
            plt.colorbar()
            plt.show()

        else:
            filters = self.encoder.weight.detach().numpy()
            filters_min = np.amin(filters)
            filters_max = np.amax(filters)

            fig, ax = plt.subplots(
                int(np.sqrt(self.encoding_dim)),
                int(np.sqrt(self.encoding_dim)),
                sharex=True,
                sharey=True,
                figsize=(8, 8),
            )
            for i in range(int(np.sqrt(self.encoding_dim))):
                for j in range(int(np.sqrt(self.encoding_dim))):
                    ax[i][j].imshow(
                        (
                            filters[i * int(np.sqrt(self.encoding_dim)) + j, 0]
                            - filters_min
                        )
                        / (filters_max - filters_min)
                    )
            # show the plot
            plt.show()
