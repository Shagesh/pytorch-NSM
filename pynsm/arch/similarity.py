"""Define a convolutional NSM module."""

import torch
from torch import nn

from typing import List, Sequence, Union, Optional

from .base import IterationLossModule


class MultiSimilarityMatching(IterationLossModule):
    """Multiple-target similarity matching circuit."""

    def __init__(
        self,
        encoders: Sequence[nn.Module],
        out_channels: int,
        tau: float = 0.1,
        max_iterations: int = 40,
        regularization: Union[str, Sequence[str]] = "weight",
        **kwargs,
    ):
        """Initialize the supervised similarity matching circuit.

        Some of the encoders can be skipped during the `forward()` call either by
        including fewer arguments than `len(encoders)` or by setting some to `None`.

        :param encoders: modules to use for encoding the inputs
        :param out_channels: number of output channels
        :param tau: factor by which to divide the competitor's learning rate
        :param max_iterations: maximum number of iterations to run in `forward()`
        :param regularization: type of encoder regularization to use; this can be a
            single string, or a sequence, to have different regularizations for each
            encoder; options are
            "weight":   use the encoders' parameters; regularization is added for all
                        the tensors returned by `encoder.parameters()`, as long as those
                        tensors are trainable (i.e., `requires_grad` is true)
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

        if isinstance(regularization, str):
            self.regularization = [regularization] * len(self.encoders)
        else:
            self.regularization = regularization

        for crt_reg in self.regularization:
            if crt_reg not in ["weight", "whiten", "none"]:
                raise ValueError(f"Unknown regularization {crt_reg}")

        self.competitor = nn.Linear(out_channels, out_channels, bias=False)
        torch.nn.init.eye_(self.competitor.weight)

        # make sure we maximize with respect to competitor weight...
        # ...and implement the learning rate ratio
        scaling = -1.0 / tau
        self.competitor.weight.register_hook(lambda g: g * scaling)

        self.y = torch.tensor([])

    def _encode(
        self, *args: Optional[torch.Tensor], keep_all: bool = False
    ) -> Sequence[torch.Tensor]:
        Wx = []
        for x, encoder in zip(args, self.encoders):
            if x is not None:
                Wx.append(encoder(x))
            elif keep_all:
                Wx.append(None)

        return Wx

    def pre_iteration(self, *args: Optional[torch.Tensor]):
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

    def post_iteration(self, *args: Optional[torch.Tensor]) -> torch.Tensor:
        super().post_iteration(*args)
        self._Wx = None
        self._Wx_sum = None

        return self.y.detach()

    def iteration_parameters(self) -> List[torch.Tensor]:
        return [self.y]

    def loss(self, *args: Optional[torch.Tensor]) -> torch.Tensor:
        y = args[-1]
        args = args[:-1]

        assert y is not None

        Wx = self._encode(*args)
        Wx_active = [_ for _ in Wx if _ is not None]
        loss = self._loss_no_reg(Wx_active, y, "mean")

        # competitor regularization
        M_reg = (self.competitor.weight**2).sum() / y.shape[1]
        loss -= M_reg

        # encoder regularization
        for encoder, regularization, crt_Wx in zip(
            self.encoders, self.regularization, Wx
        ):
            if regularization == "whiten":
                # this needs to match the scaling from _loss_no_reg!
                if crt_Wx is not None:
                    loss += 2 * (crt_Wx**2).mean()
            elif regularization == "weight":
                encoder_params = [_ for _ in encoder.parameters() if _.requires_grad]
                for weight in encoder_params:
                    loss += (weight**2).sum() * (2.0 / y.shape[1])
            elif regularization != "none":
                raise ValueError(f"Unknown regularization {regularization}")

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

    def __init__(self, out_channels: int, encoder: nn.Module, **kwargs):
        """Initialize the similarity matching circuit.

        This is a thin wrapper around `MultiSimilarityMatching` using a single target.

        :param encoder: module to use for encoding the inputs
        :param out_channels: number of output channels
        :param *args: additional positional arguments go to `MultiSimilarityMatching`
        :param **kwargs: additional keyword arguments go to `MultiSimilarityMatching`
        """
        super().__init__(encoders=[encoder], out_channels=out_channels, **kwargs)


class SupervisedSimilarityMatching(MultiSimilarityMatching):
    """Supervised similarity matching circuit for classification."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        out_channels: int,
        label_bias: bool = False,
        **kwargs,
    ):
        """Initialize the supervised similarity matching circuit.

        This is a wrapper that uses `MultiSimilarityMatching` with two encoders,
        `self.encoders == [encoder, label_encoder]`. The `label_encoder` is generated
        internally to map *floating-point* one-hot labels (which is what `forward()`
        expects) to a one-dimensional output of size `out_channels`. The `forward()`
        iteration is adapted to extend the output of the `label_encoder` to the same
        shape as the output from the `encoder()`, so it can be used in the similarity
        matching objective.

        Note that by default `"whiten"` regularization is used for the label encoder and
        `"weight"` regularization for the input encoder.

        :param encoder: module to use for encoding the inputs
        :param num_classes: number of classes for classification
        :param out_channels: number of output channels
        :param label_bias: set to true to include a bias term in the label encoder
        :param *args: additional positional arguments go to `MultiSimilarityMatching`
        :param **kwargs: additional keyword arguments go to `MultiSimilarityMatching`
        """
        label_encoder = nn.Linear(num_classes, out_channels, bias=label_bias)
        if "regularization" not in kwargs:
            kwargs["regularization"] = ("weight", "whiten")
        super().__init__(
            out_channels=out_channels, encoders=[encoder, label_encoder], **kwargs
        )

    def _encode(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        keep_all: bool = False,
    ) -> Sequence[Optional[torch.Tensor]]:
        # Wx will be shape [B, M, C1, ..., Ck], where B is batch size, M is out_channels
        Wx = self.encoders[0](x)
        if label is not None:
            # Wlabel0 will be shaoe [B, M]
            Wlabel0 = self.encoders[1](label)
            # we want to extend this to [B, M, C1, ..., Ck]
            extra_dim = Wx.ndim - 2
            Wlabel_unsqueezed = Wlabel0.reshape(Wlabel0.shape + (1,) * extra_dim)
            Wlabel = Wlabel_unsqueezed.expand(Wx.shape)

            return [Wx, Wlabel]
        else:
            if keep_all:
                return [Wx, None]
            else:
                return [Wx]
