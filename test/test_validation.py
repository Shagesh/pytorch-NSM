import pytest
from unittest.mock import Mock

import torch
from torch import nn

from typing import List, Tuple

from pynsm.util import extract_embeddings


@pytest.fixture
def linear_model() -> nn.Module:
    torch.manual_seed(21)
    return nn.Linear(5, 3)


@pytest.fixture
def data_loader() -> List[Tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(42)

    n = 12
    m = 5
    data = torch.randn(n, 5, m)
    labels = torch.randint(0, 8, (n, 5))

    loader = list(zip(data, labels))
    return loader


def test_extracts_labels_and_model_outputs(linear_model, data_loader):
    results = extract_embeddings(linear_model, data_loader)

    assert hasattr(results, "output")
    assert hasattr(results, "label")

    i = 0
    for x, label in data_loader:
        for bi in range(len(x)):
            assert torch.allclose(results.output[i], linear_model(x[bi]))
            assert results.label[i] == label[bi]
            i += 1


def test_sets_model_in_eval_mode(data_loader):
    model = Mock(return_value=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))
    model.parameters.return_value = [torch.tensor([])]
    extract_embeddings(model, data_loader)

    model.eval.assert_called_once()


def test_restores_model_to_train_mode(linear_model, data_loader):
    extract_embeddings(linear_model, data_loader)
    assert linear_model.training


def test_leaves_model_in_eval_mode_if_it_was_there(linear_model, data_loader):
    linear_model.eval()
    extract_embeddings(linear_model, data_loader)
    assert not linear_model.training


def test_progress_called_with_loader(linear_model, data_loader):
    progress = Mock(side_effect=lambda x: x)
    extract_embeddings(linear_model, data_loader, progress=progress)
    progress.assert_called_once()
