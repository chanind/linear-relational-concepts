import pytest
import torch

from multitoken_estimator.LinearRelationalEmbedding import (
    InvertedLinearRelationalEmbedding,
)


def test_InvertedLinearRelationalEmbedding_calculate_subject_activation() -> None:
    act = torch.tensor([2.0, 1.0, 1.0])
    bias = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
    weight_inv = torch.eye(3)
    inv_lre = InvertedLinearRelationalEmbedding(
        relation="test",
        subject_layer=0,
        object_layer=0,
        object_aggregation="mean",
        weight_inverse=weight_inv,
        bias=bias,
        rank=3,
    )
    vec = inv_lre.calculate_subject_activation(act, normalize=False)
    assert torch.allclose(vec, torch.tensor([1.0, 1.0, 1.0]))


def test_InvertedLinearRelationalEmbedding_calculate_subject_activation_normalize() -> (
    None
):
    act = torch.tensor([2.0, 1.0, 1.0])
    bias = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
    weight_inv = torch.eye(3)
    inv_lre = InvertedLinearRelationalEmbedding(
        relation="test",
        subject_layer=0,
        object_layer=0,
        object_aggregation="mean",
        weight_inverse=weight_inv,
        bias=bias,
        rank=3,
    )
    vec = inv_lre.calculate_subject_activation(act, normalize=True)
    assert vec.norm() == pytest.approx(1.0)
