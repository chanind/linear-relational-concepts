import torch
import torch.nn.functional as F

from multitoken_estimator.training.train_avg_concept_vector import (
    train_avg_concept_vector,
)


def test_train_avg_concept_vector() -> None:
    deltas = [
        torch.tensor([1, 2, 0.99]),
        torch.tensor([1, 2.01, 1]),
        torch.tensor([0.99, 1.99, 1.01]),
    ]
    concept_vec, metadata = train_avg_concept_vector(deltas)
    assert concept_vec.shape == (3,)
    assert torch.allclose(
        concept_vec,
        F.normalize(torch.tensor([1, 2, 1], dtype=torch.float32), dim=0),
        0.01,
    )
