import torch

from linear_relational_concepts.training.train_concept_vector_via_logistic_regression import (
    train_concept_vector_via_logistic_regression,
)


def test_train_concept_vector_via_logistic_regression() -> None:
    pos_acts = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([1.1, 0.0, 0.0]),
        torch.tensor([0.9, 0.0, 0.0]),
    ]
    neg_acts = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([-0.1, 0.0, 0.0]),
        torch.tensor([0.1, 0.0, 0.0]),
    ]
    concept_vec = train_concept_vector_via_logistic_regression(pos_acts, neg_acts)
    assert concept_vec.allclose(torch.tensor([1.0, 0.0, 0.0]), atol=0.1)
