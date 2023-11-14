import torch

from linear_relational_concepts.LinearRelationalEmbedding import (
    InvertedLinearRelationalEmbedding,
    LinearRelationalEmbedding,
)


def test_LinearRelationalEmbedding_invert() -> None:
    bias = torch.tensor([1.0, 0.0, 0.0])
    lre = LinearRelationalEmbedding(
        relation="test",
        subject_layer=5,
        object_layer=10,
        object_aggregation="mean",
        bias=bias,
        weight=torch.eye(3),
    )
    inv_lre = lre.invert(rank=2)
    assert inv_lre.relation == "test"
    assert inv_lre.subject_layer == 5
    assert inv_lre.object_layer == 10
    assert inv_lre.object_aggregation == "mean"
    assert torch.allclose(inv_lre.bias, bias)
    assert inv_lre.u.shape == (3, 2)
    assert inv_lre.s.shape == (2,)
    assert inv_lre.v.shape == (3, 2)
    assert inv_lre.rank == 2


def test_InvertedLinearRelationalEmbedding_calculate_subject_activation_unnormalized() -> (
    None
):
    acts = torch.stack(
        [
            torch.tensor([2.0, 1.0, 1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        ]
    )
    bias = torch.tensor([1.0, 0.0, 0.0])
    inv_lre = InvertedLinearRelationalEmbedding(
        relation="test",
        subject_layer=0,
        object_layer=0,
        # this u,s,v makes W_inv the identity matrix
        u=torch.eye(3),
        s=torch.ones(3),
        v=torch.eye(3),
        object_aggregation="mean",
        bias=bias,
        rank=3,
    )
    vec = inv_lre.calculate_subject_activation(acts, normalize=False)
    assert torch.allclose(vec, torch.tensor([0.5, 0.5, 0.5]))


def test_InvertedLinearRelationalEmbedding_calculate_subject_activation_normalized() -> (
    None
):
    acts = torch.stack(
        [
            torch.tensor([2.0, 1.0, 1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        ]
    )
    bias = torch.tensor([1.0, 0.0, 0.0])
    inv_lre = InvertedLinearRelationalEmbedding(
        relation="test",
        subject_layer=0,
        object_layer=0,
        # this u,s,v makes W_inv the identity matrix
        u=torch.eye(3),
        s=torch.ones(3),
        v=torch.eye(3),
        object_aggregation="mean",
        bias=bias,
        rank=3,
    )
    vec = inv_lre.calculate_subject_activation(acts, normalize=True)
    raw_target = torch.tensor([0.5, 0.5, 0.5])
    target = raw_target / raw_target.norm()
    assert torch.allclose(vec, target)
