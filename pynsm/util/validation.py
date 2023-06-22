"""Define functions to validate circuit output."""

import torch
import torch.utils
import torch.utils.data
from torch import nn

from typing import Optional, Callable
from types import SimpleNamespace


def extract_embeddings(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    progress: Optional[Callable] = None,
) -> SimpleNamespace:
    """Extract model embeddings and corresponding labels.

    This assumes a supervised-learning dataset, with samples of the form `(x, label)`.
    Note that the label is never actually used by the model.

    :param model: module to generate embeddings
    :param loader: iterable returning `(input, label)` pairs
    :param progress: progress indicator with `tqdm`-like interface
    :return: a namespace with members `output` and `label`, each a tensor containing the
        embeddings and the labels
    """
    was_training = model.training
    model.eval()

    # figure out device to send data to
    try:
        param = next(iter(model.parameters()))
        device = param.device
    except StopIteration:
        device = torch.device("cpu")

    # process the dataset
    results = SimpleNamespace(output=[], label=[])
    if progress is None:
        iterable = loader
    else:
        iterable = progress(loader)
    for x, label in iterable:
        x = x.to(device)

        y = model(x)
        results.output.append(y.detach().cpu())
        results.label.append(label.detach().cpu())

    # concatenate to tensors
    results.output = torch.cat(results.output)
    results.label = torch.cat(results.label)

    if was_training:
        model.train()

    return results
