import pytest
import torch

from multitoken_estimator.LinearRelationalEmbedding import (
    InvertedLinearRelationalEmbedding,
)


def test_InvertedLinearRelationalEmbedding_calculate_subject_activation_first_agg() -> (
    None
):
    acts = [
        torch.tensor([2.0, 1.0, 1.0]),
        torch.tensor([1.0, 0.0, 0.0]),
    ]
    bias = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
    weight_inv = torch.eye(3)
    inv_lre = InvertedLinearRelationalEmbedding(
        relation="test",
        subject_layer=0,
        object_layer=0,
        weight_inverse=weight_inv,
        bias=bias,
        rank=3,
    )
    vec = inv_lre.calculate_subject_activation(
        acts, aggregation="first", normalize=False
    )
    assert torch.allclose(vec, torch.tensor([1.0, 1.0, 1.0]))


def test_InvertedLinearRelationalEmbedding_calculate_subject_activation_mean_agg() -> (
    None
):
    acts = [
        torch.tensor([2.0, 1.0, 1.0]),
        torch.tensor([1.0, 0.0, 0.0]),
    ]
    bias = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
    weight_inv = torch.eye(3)
    inv_lre = InvertedLinearRelationalEmbedding(
        relation="test",
        subject_layer=0,
        object_layer=0,
        weight_inverse=weight_inv,
        bias=bias,
        rank=3,
    )
    vec = inv_lre.calculate_subject_activation(
        acts, aggregation="mean", normalize=False
    )
    assert torch.allclose(vec, torch.tensor([0.5, 0.5, 0.5]))


def test_InvertedLinearRelationalEmbedding_calculate_subject_activation_normalize() -> (
    None
):
    acts = [
        torch.tensor([2.0, 1.0, 1.0]),
        torch.tensor([1.0, 0.0, 0.0]),
    ]
    bias = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
    weight_inv = torch.eye(3)
    inv_lre = InvertedLinearRelationalEmbedding(
        relation="test",
        subject_layer=0,
        object_layer=0,
        weight_inverse=weight_inv,
        bias=bias,
        rank=3,
    )
    vec = inv_lre.calculate_subject_activation(acts, normalize=True)
    assert vec.norm() == pytest.approx(1.0)
