import pytest
import torch

from multitoken_estimator.LinearRelationalEmbedding import LinearRelationalEmbedding


def test_InvertedLinearRelationalEmbedding_calculate_subject_activation_normalize() -> (
    None
):
    act = torch.tensor([2.0, 1.0, 1.0])
    bias = torch.tensor([1.0, 0.0, 0.0])
    lre = LinearRelationalEmbedding(
        relation="test",
        subject_layer=0,
        object_layer=0,
        object_aggregation="mean",
        bias=bias,
        weight=torch.eye(3),
    )
    inv_lre = lre.invert(rank=3)
    vec = inv_lre.calculate_subject_activation(act, normalize=True)
    assert vec.norm() == pytest.approx(1.0)
